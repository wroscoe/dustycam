/* espnowbridge v2 — standalone ESP-NOW -> Wi-Fi repeater (battery/wall
 * powered, no host USB).
 *
 *   Joins the mesh as a STA pinned to the PRIMARY router's 2.4 GHz BSSID
 *   so the radio parks on its channel (1) — the same fixed channel the
 *   miclogger and the signal meter transmit ESP-NOW on. STA association
 *   and ESP-NOW RX then coexist on one radio.
 *
 *   Audio: reassembles miclogger "MX" chunks (meta + PCM slices),
 *   zero-fills lost slices, wraps WAV and POSTs to /blob/<dev>/audio —
 *   the same contract as the Wi-Fi path, meta gains transport/loss/rssi.
 *
 *   Meter: relays "SS" survey windows from the Heltec as MQTT readings
 *   under home/espnow/meter/... (ts corrected by each window's age).
 *
 *   Health: home/espnow/bridge/... every 30 s. OTA: /firmware/<name> like
 *   every other board — this is the last build that needs a cable.
 */
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_app_desc.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_ota_ops.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_heap_caps.h"
#include "mqtt_client.h"

static const char *TAG = "bridge2";

#define WAV_HDR 44
#define MAX_CHUNK (256 * 1024)
static size_t s_max_chunk = MAX_CHUNK;
#define SLICE_MAX 1024

static EventGroupHandle_t s_events;
#define WIFI_UP_BIT BIT0

/* one in-flight audio chunk being reassembled */
static struct {
    uint8_t *buf;              /* WAV_HDR + chunk_bytes */
    uint32_t seq, bytes, rate, ms, rms, peak, dropped;
    uint16_t n_frames, got_frames;
    int rssi_sum;
    bool have_meta;
    int64_t born_us;
} s_ch;

typedef struct {                /* recv-cb -> worker */
    uint8_t mac[6];
    int8_t rssi;
    uint16_t len;
    uint8_t data[SLICE_MAX + 16];
} pkt_t;
static QueueHandle_t s_pkt_q;

static esp_mqtt_client_handle_t s_mqtt;
static uint32_t s_frames, s_chunks, s_posted, s_meter_windows;

/* ---------------- wifi ---------------- */

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

static bool parse_mac(const char *str, uint8_t *out)
{
    unsigned m[6];
    if (sscanf(str, "%x:%x:%x:%x:%x:%x",
               &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]) != 6)
        return false;
    for (int i = 0; i < 6; i++)
        out[i] = m[i];
    return true;
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
    strlcpy((char *)sta.sta.ssid, CONFIG_BR_WIFI_SSID,
            sizeof(sta.sta.ssid));
    strlcpy((char *)sta.sta.password, CONFIG_BR_WIFI_PASS,
            sizeof(sta.sta.password));
    /* pin to the primary router: keeps the radio on the fixed ESP-NOW
     * channel; a satellite on another channel would break reception */
    if (parse_mac(CONFIG_BR_AP_BSSID, sta.sta.bssid))
        sta.sta.bssid_set = 1;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));   /* RX must not doze */
    /* LR added alongside bgn: AP link stays 11n, but the radio can now
     * decode the miclogger's long-range ESP-NOW frames (~+7 dB budget) */
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

/* ---------------- upload + ota (fleet pattern) ---------------- */

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

static bool post_chunk(void)
{
    uint16_t lost = s_ch.n_frames - s_ch.got_frames;
    char url[160], meta[300];
    snprintf(url, sizeof(url), "http://%s:%d/blob/%s/audio",
             CONFIG_BR_SERVER_HOST, CONFIG_BR_SERVER_PORT,
             CONFIG_BR_MIC_DEVICE);
    snprintf(meta, sizeof(meta),
             "{\"seq\":%lu,\"rate\":%lu,\"ms\":%lu,\"rms\":%lu,"
             "\"peak\":%lu,\"dropped\":%lu,\"transport\":\"espnow\","
             "\"loss_frames\":%u,\"n_frames\":%u,\"rssi\":%d,"
             "\"via\":\"%s\"}",
             (unsigned long)s_ch.seq, (unsigned long)s_ch.rate,
             (unsigned long)s_ch.ms, (unsigned long)s_ch.rms,
             (unsigned long)s_ch.peak, (unsigned long)s_ch.dropped,
             lost, s_ch.n_frames,
             s_ch.got_frames ? s_ch.rssi_sum / (int)s_ch.got_frames : 0,
             esp_app_get_description()->version);
    esp_http_client_config_t hc = {
        .url = url, .method = HTTP_METHOD_POST, .timeout_ms = 8000,
    };
    esp_http_client_handle_t h = esp_http_client_init(&hc);
    if (!h) return false;
    esp_http_client_set_header(h, "Content-Type", "audio/wav");
    esp_http_client_set_header(h, "X-Meta", meta);
    bool ok = false;
    size_t len = WAV_HDR + s_ch.bytes;
    if (esp_http_client_open(h, len) == ESP_OK) {
        if (esp_http_client_write(h, (const char *)s_ch.buf, len)
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
             CONFIG_BR_SERVER_HOST, CONFIG_BR_SERVER_PORT,
             CONFIG_BR_DEVICE_NAME);
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
             CONFIG_BR_SERVER_HOST, CONFIG_BR_SERVER_PORT,
             CONFIG_BR_DEVICE_NAME);
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

/* ---------------- wav + reassembly ---------------- */

static void wav_header(uint8_t *h, uint32_t pcm, uint32_t rate)
{
    uint32_t v; uint16_t u;
    memcpy(h, "RIFF", 4); v = 36 + pcm; memcpy(h + 4, &v, 4);
    memcpy(h + 8, "WAVEfmt ", 8); v = 16; memcpy(h + 16, &v, 4);
    u = 1; memcpy(h + 20, &u, 2); memcpy(h + 22, &u, 2);
    memcpy(h + 24, &rate, 4); v = rate * 2; memcpy(h + 28, &v, 4);
    u = 2; memcpy(h + 32, &u, 2); u = 16; memcpy(h + 34, &u, 2);
    memcpy(h + 36, "data", 4); memcpy(h + 40, &pcm, 4);
}

static void chunk_flush(void)
{
    if (!s_ch.have_meta || !s_ch.got_frames)
        goto reset;
    wav_header(s_ch.buf, s_ch.bytes, s_ch.rate);
    if (wifi_wait(8000) && post_chunk()) {
        mark_valid_once();
        s_posted++;
    }
    s_chunks++;
reset:
    s_ch.have_meta = false;
    s_ch.got_frames = 0;
    s_ch.rssi_sum = 0;
}

static void handle_mx(const pkt_t *p)
{
    if (p->len < 11) return;
    uint8_t typ = p->data[2];
    uint32_t seq; uint16_t idx, n;
    memcpy(&seq, p->data + 3, 4);
    memcpy(&idx, p->data + 7, 2);
    memcpy(&n, p->data + 9, 2);
    const uint8_t *body = p->data + 11;
    uint16_t blen = p->len - 11;

    if (s_ch.have_meta && seq != s_ch.seq)
        chunk_flush();
    if (typ == 1 && blen >= 24) {
        uint32_t m[6];
        memcpy(m, body, 24);
        if (m[5] == 0 || m[5] > s_max_chunk - WAV_HDR)
            return;
        s_ch.rate = m[0]; s_ch.ms = m[1]; s_ch.rms = m[2];
        s_ch.peak = m[3]; s_ch.dropped = m[4]; s_ch.bytes = m[5];
        s_ch.seq = seq; s_ch.n_frames = n;
        s_ch.have_meta = true;
        s_ch.born_us = esp_timer_get_time();
        memset(s_ch.buf + WAV_HDR, 0, s_ch.bytes);
    } else if (typ == 0 && s_ch.have_meta && seq == s_ch.seq && idx >= 1) {
        size_t off = (size_t)(idx - 1) * 1000;   /* miclogger slice size */
        if (off + blen <= s_ch.bytes) {
            memcpy(s_ch.buf + WAV_HDR + off, body, blen);
            s_ch.got_frames++;
            s_ch.rssi_sum += p->rssi;
        }
        if (s_ch.got_frames >= s_ch.n_frames)
            chunk_flush();
    }
}

/* ---------------- meter relay ---------------- */

static void handle_ss(const pkt_t *p)
{
    /* "SS" | ver u8 | count u8 | count x { age_s u16, frames u16,
     * rssi_avg i8, rssi_max i8 } */
    if (p->len < 4 || !s_mqtt) return;
    uint8_t cnt = p->data[3];
    const uint8_t *w = p->data + 4;
    int64_t now = 0;   /* broker/ingest stamps arrival; we send age-corrected offsets only when we have SNTP — we don't, so publish v + age */
    (void)now;
    for (int i = 0; i < cnt && 4 + (i + 1) * 6 <= p->len; i++, w += 6) {
        uint16_t age_s, frames; int8_t avg, mx;
        memcpy(&age_s, w, 2);
        memcpy(&frames, w + 2, 2);
        avg = (int8_t)w[4]; mx = (int8_t)w[5];
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "{\"v\":%d,\"max\":%d,\"frames\":%u,\"age_s\":%u,"
                 "\"meter\":\"%02x%02x%02x\"}",
                 avg, mx, frames, age_s,
                 p->mac[3], p->mac[4], p->mac[5]);
        esp_mqtt_client_publish(s_mqtt, "home/espnow/meter/rssi", msg,
                                0, 0, 0);
        s_meter_windows++;
    }
}

/* ---------------- espnow rx ---------------- */

static void recv_cb(const esp_now_recv_info_t *info, const uint8_t *data,
                    int len)
{
    if (len < 2 || len > (int)sizeof(((pkt_t *)0)->data))
        return;
    pkt_t p;
    memcpy(p.mac, info->src_addr, 6);
    p.rssi = info->rx_ctrl ? info->rx_ctrl->rssi : 0;
    p.len = len;
    memcpy(p.data, data, len);
    if (xQueueSend(s_pkt_q, &p, 0) == pdTRUE)
        s_frames++;
}

static void worker_task(void *arg)
{
    pkt_t p;
    int64_t last_health = 0, last_ota = 0;
    for (;;) {
        if (xQueueReceive(s_pkt_q, &p, pdMS_TO_TICKS(1000)) == pdTRUE) {
            if (p.data[0] == 'M' && p.data[1] == 'X')
                handle_mx(&p);
            else if (p.data[0] == 'S' && p.data[1] == 'S')
                handle_ss(&p);
        }
        int64_t now = esp_timer_get_time();
        /* half-finished chunk with a silent sender: flush what we have */
        if (s_ch.have_meta && now - s_ch.born_us > 15 * 1000000)
            chunk_flush();
        if (now - last_health > 30 * 1000000) {
            last_health = now;
            if (s_mqtt) {
                char msg[96];
                snprintf(msg, sizeof(msg),
                         "{\"v\":%lu,\"chunks\":%lu,\"posted\":%lu,"
                         "\"meter_w\":%lu}",
                         (unsigned long)s_frames, (unsigned long)s_chunks,
                         (unsigned long)s_posted,
                         (unsigned long)s_meter_windows);
                esp_mqtt_client_publish(s_mqtt, "home/espnow/bridge/frames",
                                        msg, 0, 0, 0);
            }
        }
        if (now - last_ota > 60 * 1000000) {
            last_ota = now;
            if (wifi_wait(0))
                check_ota_once();
        }
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
    s_pkt_q = xQueueCreate(48, sizeof(pkt_t));
    s_ch.buf = heap_caps_malloc(MAX_CHUNK, MALLOC_CAP_SPIRAM);
    if (!s_ch.buf) {
        /* no PSRAM: internal RAM holds one 16 kHz 5 s chunk, barely */
        s_ch.buf = malloc(170 * 1024);
        s_max_chunk = 170 * 1024;
        ESP_LOGW(TAG, "no PSRAM, internal chunk buf");
    }
    if (!s_ch.buf) {
        ESP_LOGE(TAG, "chunk buf alloc failed");
        abort();
    }

    wifi_start();
    if (!wifi_wait(30000))
        ESP_LOGW(TAG, "wifi slow; espnow rx waits for association");
    else
        ESP_LOGI(TAG, "wifi up");

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_recv_cb(recv_cb));

    char uri[128];
    snprintf(uri, sizeof(uri), "mqtt://%s:1883", CONFIG_BR_SERVER_HOST);
    esp_mqtt_client_config_t mc = {
        .broker.address.uri = uri,
        .credentials.username = "espnow",
        .credentials.authentication.password = CONFIG_BR_MQTT_PASS,
    };
    s_mqtt = esp_mqtt_client_init(&mc);
    esp_mqtt_client_start(s_mqtt);

    xTaskCreate(worker_task, "worker", 8192, NULL, 10, NULL);
    ESP_LOGI(TAG, "%s up", esp_app_get_description()->version);
}
