/* miclogger — Seeed XIAO ESP32S3 Sense: continuous PDM mic -> sensorhub.
 *
 *   Capture task: I2S PDM RX (onboard MSM261 mic, CLK=42 DIN=41) fills a
 *   PSRAM ring of WAV chunks. Upload task POSTs each chunk to
 *   /blob/<device>/audio with X-Meta (seq, rms, peak, dropped, rssi, fw)
 *   and polls /firmware/<device>/version for OTA every OTA_POLL_S.
 *
 *   The ring is the wireless-jitter cushion (RING_CHUNKS * CHUNK_S seconds);
 *   when uploads stall past that, whole chunks are dropped and counted in
 *   the next chunk's meta rather than silently lost.
 *
 *   USB-powered, always on: no deep sleep, unlike camlogger.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_app_desc.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_ota_ops.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_sleep.h"
#include "nvs_flash.h"
#include "driver/i2s_pdm.h"
#include "esp_now.h"
#include "freertos/semphr.h"

static const char *TAG = "miclogger";

#define MIC_GPIO_CLK  GPIO_NUM_42
#define MIC_GPIO_DIN  GPIO_NUM_41

#define RATE        CONFIG_MIC_SAMPLE_RATE
#define CHUNK_S     CONFIG_MIC_CHUNK_S
#define RING_CHUNKS CONFIG_MIC_RING_CHUNKS
#define OTA_POLL_S  30

#define WAV_HDR     44
#define CHUNK_SAMPLES (RATE * CHUNK_S)
#define CHUNK_BYTES   (CHUNK_SAMPLES * 2)

typedef struct {
    uint8_t *buf;              /* WAV_HDR + CHUNK_BYTES, header prefilled */
    uint32_t seq;
    uint32_t rms, peak, dropped;
} chunk_t;

static chunk_t s_ring[RING_CHUNKS];
static QueueHandle_t s_free_q, s_full_q;   /* carry ring indexes */

static EventGroupHandle_t s_events;
#define WIFI_UP_BIT BIT0

/* ---------------- wifi (camlogger pattern) ---------------- */

static volatile bool s_no_reconnect;   /* espnow mode: leave channel alone */

static void wifi_event(void *arg, esp_event_base_t base, int32_t id,
                       void *data)
{
    if (s_no_reconnect)
        return;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_events, WIFI_UP_BIT);
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_events, WIFI_UP_BIT);
    }
}

static void wifi_start(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t wcfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wcfg));
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                        wifi_event, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                        wifi_event, NULL, NULL);
    wifi_config_t sta = { 0 };
    strlcpy((char *)sta.sta.ssid, CONFIG_MIC_WIFI_SSID,
            sizeof(sta.sta.ssid));
    strlcpy((char *)sta.sta.password, CONFIG_MIC_WIFI_PASS,
            sizeof(sta.sta.password));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
    ESP_ERROR_CHECK(esp_wifi_start());
    /* PS_NONE keeps draw ~100mA+: below ~50mA the USB pack's auto-
     * shutoff latches off within 45s (measured) — modem-sleep between
     * posts was dipping into exactly that zone */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    /* LR alongside bgn: normal AP association, long-range ESP-NOW */
    ESP_ERROR_CHECK(esp_wifi_set_protocol(
        WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G |
        WIFI_PROTOCOL_11N | WIFI_PROTOCOL_LR));
}

static bool wifi_wait(int timeout_ms)
{
    return (xEventGroupWaitBits(s_events, WIFI_UP_BIT, pdFALSE, pdTRUE,
                                pdMS_TO_TICKS(timeout_ms)) & WIFI_UP_BIT)
           != 0;
}

static int wifi_rssi(void)
{
    wifi_ap_record_t ap;
    return esp_wifi_sta_get_ap_info(&ap) == ESP_OK ? ap.rssi : 0;
}

/* "aabbccddeeff/6" — associated AP + channel, for mesh-roaming forensics */
static const char *wifi_bssid(void)
{
    static char s[20];
    wifi_ap_record_t ap;
    if (esp_wifi_sta_get_ap_info(&ap) != ESP_OK)
        return "?";
    snprintf(s, sizeof(s), "%02x%02x%02x%02x%02x%02x/%d",
             ap.bssid[0], ap.bssid[1], ap.bssid[2],
             ap.bssid[3], ap.bssid[4], ap.bssid[5], ap.primary);
    return s;
}

/* ---------------- wav ---------------- */

static void wav_header(uint8_t *h, uint32_t pcm_bytes, uint32_t rate)
{
    uint32_t byterate = rate * 2;
    memcpy(h, "RIFF", 4);
    uint32_t v = 36 + pcm_bytes;          memcpy(h + 4, &v, 4);
    memcpy(h + 8, "WAVEfmt ", 8);
    v = 16;                               memcpy(h + 16, &v, 4);
    uint16_t u = 1;                       memcpy(h + 20, &u, 2); /* PCM  */
    u = 1;                                memcpy(h + 22, &u, 2); /* mono */
    memcpy(h + 24, &rate, 4);
    memcpy(h + 28, &byterate, 4);
    u = 2;                                memcpy(h + 32, &u, 2);
    u = 16;                               memcpy(h + 34, &u, 2);
    memcpy(h + 36, "data", 4);
    memcpy(h + 40, &pcm_bytes, 4);
}

/* ---------------- capture ---------------- */

static i2s_chan_handle_t s_rx;

static void mic_init(void)
{
    /* PDM RX exists only on I2S0 on the S3 */
    i2s_chan_config_t cc =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &s_rx));
    i2s_pdm_rx_config_t pc = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = { .clk = MIC_GPIO_CLK, .din = MIC_GPIO_DIN },
    };
    /* PDM mic clock = rate x downsample. The default 64x ratio puts an
     * 8 kHz rate at 512 kHz mic clock -- below the MSM261's ~1 MHz
     * minimum, and it degrades to broadband hiss. 128x restores
     * 1.024 MHz at 8 kHz. */
    if (RATE <= 8000)
        pc.clk_cfg.dn_sample_mode = I2S_PDM_DSR_16S;
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(s_rx, &pc));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx));
}

static void fill_chunk(chunk_t *c, uint32_t seq, uint32_t dropped)
{
        int16_t *pcm = (int16_t *)(c->buf + WAV_HDR);
        size_t got = 0;
        while (got < CHUNK_BYTES) {
            size_t n = 0;
            esp_err_t err = i2s_channel_read(
                s_rx, (uint8_t *)pcm + got, CHUNK_BYTES - got, &n,
                pdMS_TO_TICKS(2000));
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "i2s read: %s", esp_err_to_name(err));
                vTaskDelay(pdMS_TO_TICKS(100));
            }
            got += n;
        }
        /* raw PDM RX sits on a large DC offset with the signal ~60 dB
         * down: remove per-chunk DC, then fixed digital gain w/ clamp */
        int64_t sum = 0;
        for (int i = 0; i < CHUNK_SAMPLES; i++) sum += pcm[i];
        int32_t dc = sum / CHUNK_SAMPLES;
        uint64_t sum2 = 0;
        uint32_t peak = 0;
        for (int i = 0; i < CHUNK_SAMPLES; i++) {
            int32_t s = (pcm[i] - dc) << CONFIG_MIC_GAIN_SHIFT;
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            pcm[i] = s;
            sum2 += (int64_t)s * s;
            uint32_t a = s < 0 ? -s : s;
            if (a > peak) peak = a;
        }
        c->seq = seq;
        c->rms = (uint32_t)sqrt((double)sum2 / CHUNK_SAMPLES);
        c->peak = peak;
        c->dropped = dropped;
}

static void capture_task(void *arg)
{
    uint32_t seq = 0, dropped = 0;
    for (;;) {
        int idx;
        if (xQueueReceive(s_free_q, &idx, 0) != pdTRUE) {
            /* uploads stalled past the ring: sacrifice the oldest chunk */
            if (xQueueReceive(s_full_q, &idx, portMAX_DELAY) == pdTRUE)
                dropped++;
            else
                continue;
        }
        fill_chunk(&s_ring[idx], seq++, dropped);
        xQueueSend(s_full_q, &idx, 0);
    }
}

/* ---------------- upload + ota (camlogger pattern) ---------------- */

/* extra JSON fields (leading comma) appended to X-Meta; power probe only */
static char s_extra_meta[64] = "";

static void mark_valid_once(void)
{
    static bool done;
    if (done) return;
    esp_ota_img_states_t st;
    const esp_partition_t *run = esp_ota_get_running_partition();
    if (esp_ota_get_state_partition(run, &st) == ESP_OK &&
        st == ESP_OTA_IMG_PENDING_VERIFY)
        esp_ota_mark_app_valid_cancel_rollback();
    done = true;
}

static bool post_wav(const chunk_t *c)
{
    char url[160], meta[300];
    snprintf(url, sizeof(url), "http://%s:%d/blob/%s/audio",
             CONFIG_MIC_SERVER_HOST, CONFIG_MIC_SERVER_PORT,
             CONFIG_MIC_DEVICE_NAME);
    snprintf(meta, sizeof(meta),
             "{\"seq\":%lu,\"rate\":%d,\"ms\":%d,\"rms\":%lu,"
             "\"peak\":%lu,\"dropped\":%lu,\"rssi\":%d,"
             "\"bssid\":\"%s\",\"fw\":\"%s\"%s}",
             (unsigned long)c->seq, RATE, CHUNK_S * 1000,
             (unsigned long)c->rms, (unsigned long)c->peak,
             (unsigned long)c->dropped, wifi_rssi(), wifi_bssid(),
             esp_app_get_description()->version, s_extra_meta);
    esp_http_client_config_t hc = {
        .url = url, .method = HTTP_METHOD_POST, .timeout_ms = 8000,
    };
    esp_http_client_handle_t h = esp_http_client_init(&hc);
    if (!h) return false;
    esp_http_client_set_header(h, "Content-Type", "audio/wav");
    esp_http_client_set_header(h, "X-Meta", meta);
    bool ok = false;
    size_t len = WAV_HDR + CHUNK_BYTES;
    if (esp_http_client_open(h, len) == ESP_OK) {
        if (esp_http_client_write(h, (const char *)c->buf, len)
            == (int)len) {
            esp_http_client_fetch_headers(h);
            int st = esp_http_client_get_status_code(h);
            ok = st >= 200 && st < 300;
        }
    }
    esp_http_client_close(h);
    esp_http_client_cleanup(h);
    return ok;
}

static void check_ota_once(void)
{
    char url[160], remote[36];
    const char *cur = esp_app_get_description()->version;
    snprintf(url, sizeof(url), "http://%s:%d/firmware/%s/version",
             CONFIG_MIC_SERVER_HOST, CONFIG_MIC_SERVER_PORT,
             CONFIG_MIC_DEVICE_NAME);
    esp_http_client_config_t hc = { .url = url, .timeout_ms = 5000 };
    esp_http_client_handle_t c = esp_http_client_init(&hc);
    int n = 0;
    if (c && esp_http_client_open(c, 0) == ESP_OK) {
        esp_http_client_fetch_headers(c);
        if (esp_http_client_get_status_code(c) == 200)
            n = esp_http_client_read(c, remote, sizeof(remote) - 1);
    }
    if (c) { esp_http_client_close(c); esp_http_client_cleanup(c); }
    if (n <= 0) return;
    remote[n] = 0;
    while (n > 0 && (remote[n-1] == '\n' || remote[n-1] == '\r'))
        remote[--n] = 0;
    if (!strlen(remote) || strcmp(remote, cur) == 0) return;

    ESP_LOGW(TAG, "OTA %s -> %s", cur, remote);
    snprintf(url, sizeof(url), "http://%s:%d/firmware/%s.bin",
             CONFIG_MIC_SERVER_HOST, CONFIG_MIC_SERVER_PORT,
             CONFIG_MIC_DEVICE_NAME);
    esp_http_client_config_t oc = {
        .url = url, .timeout_ms = 15000, .keep_alive_enable = true,
    };
    esp_https_ota_config_t ota = { .http_config = &oc };
    if (esp_https_ota(&ota) == ESP_OK) {
        ESP_LOGW(TAG, "OTA ok, rebooting");
        esp_restart();
    }
    ESP_LOGE(TAG, "OTA failed");
}

/* Post chunks over HTTP until wifi looks dead: 6 consecutive failures.
 * Chunks are never retried — the ring is the buffer, and stale audio is
 * worth less than fresh audio. */
static void wifi_phase(void)
{
    int64_t last_ota = 0;
    int fails = 0;
    while (fails < 6) {
        int idx;
        if (xQueueReceive(s_full_q, &idx, pdMS_TO_TICKS(10000)) != pdTRUE)
            continue;
        chunk_t *c = &s_ring[idx];
        bool sent = wifi_wait(8000) && post_wav(c);
        if (sent) {
            mark_valid_once();
            fails = 0;
        } else {
            fails++;
            ESP_LOGW(TAG, "chunk %lu upload failed (%d consecutive)",
                     (unsigned long)c->seq, fails);
        }
        xQueueSend(s_free_q, &idx, 0);

        int64_t now = esp_timer_get_time();
        if (sent && now - last_ota > (int64_t)OTA_POLL_S * 1000000) {
            last_ota = now;
            check_ota_once();
        }
    }
}

/* ---------------- power-bank probe ----------------
 * One-shot experiment: step from continuous draw into ever longer
 * light-sleep gaps between capture+post bursts, tagging every post with
 * the current gap. The stage in the last post received before the board
 * goes silent = the longest idle the pack's auto-shutoff tolerates.
 * NVS-gated per firmware version, so a power-loss reboot (the expected
 * outcome!) does not re-run it — the board falls through to normal
 * continuous streaming instead. */

#if CONFIG_MIC_POWER_PROBE
static const int PROBE_SLEEPS[] = {0, 5, 10, 20, 30, 45, 60, 90,
                                   120, 180, 300};
#define PROBE_CYCLES 6           /* bursts per stage */

static bool probe_pending(void)
{
    nvs_handle_t h;
    char tok[40] = "";
    size_t sz = sizeof(tok);
    const char *cur = esp_app_get_description()->version;
    if (nvs_open("miccfg", NVS_READWRITE, &h) != ESP_OK)
        return false;
    esp_err_t e = nvs_get_str(h, "probe_tok", tok, &sz);
    bool run = (e != ESP_OK) || strcmp(tok, cur) != 0;
    if (run) {
        nvs_set_str(h, "probe_tok", cur);
        nvs_commit(h);
    }
    nvs_close(h);
    return run;
}

static void power_probe(void)
{
    chunk_t *c = &s_ring[0];
    uint32_t seq = 0;
    ESP_LOGW(TAG, "POWER PROBE: %d stages x %d bursts",
             (int)(sizeof(PROBE_SLEEPS) / sizeof(*PROBE_SLEEPS)),
             PROBE_CYCLES);
    for (int st = 0; st < sizeof(PROBE_SLEEPS) / sizeof(*PROBE_SLEEPS);
         st++) {
        int gap = PROBE_SLEEPS[st];
        for (int cy = 0; cy < PROBE_CYCLES; cy++) {
            snprintf(s_extra_meta, sizeof(s_extra_meta),
                     ",\"probe_gap\":%d,\"probe_cycle\":%d", gap, cy);
            fill_chunk(c, seq++, 0);
            if (wifi_wait(15000)) {
                if (post_wav(c))
                    mark_valid_once();
                if (cy == 0)
                    check_ota_once();   /* remote abort hatch */
            }
            if (gap > 0) {
                i2s_channel_disable(s_rx);
                esp_wifi_stop();
                esp_sleep_enable_timer_wakeup((uint64_t)gap * 1000000ULL);
                esp_light_sleep_start();
                esp_wifi_start();
                i2s_channel_enable(s_rx);
            }
        }
        ESP_LOGW(TAG, "probe: survived gap=%ds", gap);
    }
    s_extra_meta[0] = 0;
    ESP_LOGW(TAG, "probe complete, resuming continuous streaming");
}
#endif /* CONFIG_MIC_POWER_PROBE */

/* ---------------- transport state machine ----------------
 * Wi-Fi is preferred (OTA lives there). When it is unreachable, stream
 * the same chunks over ESP-NOW *unicast* to the espnowbridge devkit --
 * unicast so every frame is MAC-ACKed and delivery is measurable. The
 * loop alternates: wifi dead -> espnow; espnow dead (3 chunks with ~no
 * ACKed frames) -> retry wifi; a healthy espnow link still retries
 * wifi every 15 min so an upstairs board finds its way back to OTA.
 * Payload: "MX" | type u8 (0 data, 1 meta) | chunk_seq u32 | frame_idx
 * u16 | n_frames u16 | body. */

#define ESPNOW_CHANNEL        1
#define ESPNOW_SLICE          1000
#define ESPNOW_DEAD_CHUNKS    3
#define WIFI_RETRY_PERIOD_US  (15LL * 60 * 1000000)
#define WIFI_JOIN_WAIT_MS     45000
#define WIFI_REJOIN_WAIT_MS   30000

static SemaphoreHandle_t s_now_sent;
static bool s_use_lr = true;      /* alternates on totally-dead phases */
static volatile bool s_last_ack;
static uint8_t s_bridge_mac[6];

static void now_send_cb(const esp_now_send_info_t *info,
                        esp_now_send_status_t status)
{
    s_last_ack = status == ESP_NOW_SEND_SUCCESS;
    xSemaphoreGive(s_now_sent);
}

/* send until the bridge MAC-ACKs, up to 3 transmissions; true = ACKed.
 * Retransmits turned a measured 45% frame loss at -86 dBm into the
 * exponent: p^3 residual instead of p. */
static bool now_send(const uint8_t *buf, size_t len)
{
    for (int try = 0; try < 3; try++) {
        if (esp_now_send(s_bridge_mac, buf, len) != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(5));   /* TX queue full: back off */
            continue;
        }
        s_last_ack = false;
        xSemaphoreTake(s_now_sent, pdMS_TO_TICKS(150));
        if (s_last_ack)
            return true;
    }
    return false;
}

static void espnow_start(void)
{
    s_no_reconnect = true;
    esp_wifi_disconnect();
    vTaskDelay(pdMS_TO_TICKS(500));   /* let any in-flight scan finish */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_channel(ESPNOW_CHANNEL,
                                         WIFI_SECOND_CHAN_NONE));
    ESP_ERROR_CHECK(esp_now_init());
    if (!s_now_sent)
        s_now_sent = xSemaphoreCreateBinary();
    esp_now_register_send_cb(now_send_cb);
    esp_now_peer_info_t peer = { 0 };
    memcpy(peer.peer_addr, s_bridge_mac, 6);
    peer.channel = ESPNOW_CHANNEL;
    peer.ifidx = WIFI_IF_STA;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    /* LR 250K buys ~+7 dB; if the whole previous espnow phase got zero
     * ACKs, alternate back to the stock 1M rate — covers any LR quirk
     * on either end without stranding the board. Never fatal. */
    esp_err_t rerr;
    if (s_use_lr) {
        esp_now_rate_config_t rc = {
            .phymode = WIFI_PHY_MODE_LR,
            .rate = WIFI_PHY_RATE_LORA_250K,
        };
        rerr = esp_now_set_peer_rate_config(s_bridge_mac, &rc);
    } else {
        esp_now_rate_config_t rc = {
            .phymode = WIFI_PHY_MODE_11B,
            .rate = WIFI_PHY_RATE_1M_L,
        };
        rerr = esp_now_set_peer_rate_config(s_bridge_mac, &rc);
    }
    if (rerr != ESP_OK)
        ESP_LOGW(TAG, "rate config (%s): %s", s_use_lr ? "LR" : "1M",
                 esp_err_to_name(rerr));
    ESP_LOGW(TAG, "espnow rate: %s", s_use_lr ? "LR-250K" : "11b-1M");
}

static void espnow_stop(void)
{
    esp_now_deinit();
    s_no_reconnect = false;
}

/* true if the bridge ACKed a useful share of the chunk's frames */
static bool espnow_send_chunk(const chunk_t *c)
{
    static uint8_t frame[11 + ESPNOW_SLICE];
    frame[0] = 'M'; frame[1] = 'X';
    uint16_t n = (CHUNK_BYTES + ESPNOW_SLICE - 1) / ESPNOW_SLICE;
    uint32_t delivered = 0;
    uint32_t meta[6] = { RATE, CHUNK_S * 1000, c->rms, c->peak,
                         c->dropped, CHUNK_BYTES };
    frame[2] = 1;
    memcpy(frame + 3, &c->seq, 4);
    memset(frame + 7, 0, 2);
    memcpy(frame + 9, &n, 2);
    memcpy(frame + 11, meta, sizeof(meta));
    now_send(frame, 11 + sizeof(meta));
    frame[2] = 0;
    for (uint16_t i = 0; i < n; i++) {
        uint16_t fi = i + 1;
        size_t off = (size_t)i * ESPNOW_SLICE;
        size_t len = CHUNK_BYTES - off;
        if (len > ESPNOW_SLICE) len = ESPNOW_SLICE;
        memcpy(frame + 7, &fi, 2);
        memcpy(frame + 11, c->buf + WAV_HDR + off, len);
        if (now_send(frame, 11 + len))
            delivered++;
    }
    return delivered > n / 4;
}

static void transport_task(void *arg)
{
    bool first = true;
    for (;;) {
        /* ---- Wi-Fi phase (preferred: OTA lives here) ---- */
        s_no_reconnect = false;
        esp_wifi_connect();
        if (wifi_wait(first ? WIFI_JOIN_WAIT_MS : WIFI_REJOIN_WAIT_MS)) {
            ESP_LOGW(TAG, "transport: wifi via %s", wifi_bssid());
            wifi_phase();                 /* returns when wifi is dead */
            ESP_LOGW(TAG, "wifi lost");
        }
        first = false;
        /* ---- ESP-NOW phase ---- */
        espnow_start();
        ESP_LOGW(TAG, "transport: espnow ch %d", ESPNOW_CHANNEL);
        int dead = 0;
        int64_t entered = esp_timer_get_time();
        while (dead < ESPNOW_DEAD_CHUNKS &&
               esp_timer_get_time() - entered < WIFI_RETRY_PERIOD_US) {
            int idx;
            if (xQueueReceive(s_full_q, &idx, pdMS_TO_TICKS(10000))
                != pdTRUE)
                continue;
            bool ok = espnow_send_chunk(&s_ring[idx]);
            xQueueSend(s_free_q, &idx, 0);
            if (ok) {
                mark_valid_once();
                dead = 0;
            } else {
                dead++;
            }
        }
        espnow_stop();
        if (dead >= ESPNOW_DEAD_CHUNKS)
            s_use_lr = !s_use_lr;   /* nothing ACKed all phase: try the
                                     * other rate next time */
        ESP_LOGW(TAG, "%s", dead >= ESPNOW_DEAD_CHUNKS
                 ? "espnow dead: retrying wifi" : "periodic wifi retry");
    }
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
        err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }
    s_events = xEventGroupCreate();

    s_free_q = xQueueCreate(RING_CHUNKS, sizeof(int));
    s_full_q = xQueueCreate(RING_CHUNKS, sizeof(int));
    for (int i = 0; i < RING_CHUNKS; i++) {
        s_ring[i].buf = heap_caps_malloc(WAV_HDR + CHUNK_BYTES,
                                         MALLOC_CAP_SPIRAM);
        if (!s_ring[i].buf) {
            ESP_LOGE(TAG, "PSRAM alloc %d failed", i);
            abort();
        }
        wav_header(s_ring[i].buf, CHUNK_BYTES, RATE);
        xQueueSend(s_free_q, &i, 0);
    }
    ESP_LOGI(TAG, "%s: %d Hz, %ds chunks (%d KB), ring %d (%ds cushion)",
             esp_app_get_description()->version, RATE, CHUNK_S,
             CHUNK_BYTES / 1024, RING_CHUNKS, RING_CHUNKS * CHUNK_S);

    wifi_start();
    mic_init();
#if CONFIG_MIC_POWER_PROBE
    if (probe_pending())
        power_probe();
#endif
    unsigned m[6];
    if (sscanf(CONFIG_MIC_BRIDGE_MAC, "%x:%x:%x:%x:%x:%x",
               &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]) == 6)
        for (int i = 0; i < 6; i++)
            s_bridge_mac[i] = m[i];

    xTaskCreatePinnedToCore(capture_task, "capture", 4096, NULL, 10, NULL, 1);
    xTaskCreatePinnedToCore(transport_task, "transport", 8192, NULL, 5,
                            NULL, 0);
}
