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

static void wifi_event(void *arg, esp_event_base_t base, int32_t id,
                       void *data)
{
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
             "\"peak\":%lu,\"dropped\":%lu,\"rssi\":%d,\"fw\":\"%s\"%s}",
             (unsigned long)c->seq, RATE, CHUNK_S * 1000,
             (unsigned long)c->rms, (unsigned long)c->peak,
             (unsigned long)c->dropped, wifi_rssi(),
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

static void upload_task(void *arg)
{
    int64_t last_ota = 0;
    for (;;) {
        int idx;
        if (xQueueReceive(s_full_q, &idx, portMAX_DELAY) != pdTRUE)
            continue;
        chunk_t *c = &s_ring[idx];
        bool sent = false;
        if (wifi_wait(10000)) {
            sent = post_wav(c);
            if (sent)
                mark_valid_once();
        }
        if (!sent)
            ESP_LOGW(TAG, "chunk %lu upload failed (ring holds %d)",
                     (unsigned long)c->seq,
                     (int)uxQueueMessagesWaiting(s_full_q));
        /* failed chunk is not retried: the ring itself is the buffer,
         * and stale audio is worth less than fresh audio */
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
    if (probe_pending())
        power_probe();
    xTaskCreatePinnedToCore(capture_task, "capture", 4096, NULL, 10, NULL, 1);
    xTaskCreatePinnedToCore(upload_task, "upload", 8192, NULL, 5, NULL, 0);
}
