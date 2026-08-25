/* camlogger v2: deep-sleep motion-gated frame logger (Phase 1).
 *
 * Wake cycle (every CAM_WAKE_INTERVAL_S):
 *   wake -> capture JPEG -> tiny grayscale thumbnail (1/8 decode + box
 *   downsample) -> diff against previous thumbnail in RTC memory ->
 *   motion? (or heartbeat due / first boot / pending OTA verify) ->
 *   WiFi up, POST frame, check OTA -> deep sleep.
 * No motion: back to sleep without ever touching the radio (the savings).
 *
 * Safety (learned the hard way, see sensorhub + wavesharecam LESSONS.md):
 *  - Deep-sleep wake IS a reset: a pending-verify OTA image MUST transmit
 *    and mark itself valid before its first sleep, or the bootloader
 *    rolls it back. If it can't upload, we reboot -> rollback = correct.
 *  - Hold BOOT during a wake to drop into always-on recovery mode
 *    (continuous 5s frames + 30s OTA polling) for easy iteration.
 *  - Camera config zero-initialized; warm-up frames discarded.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "esp_app_desc.h"
#include "esp_camera.h"
#include "esp_event.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_ota_ops.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "img_converters.h"
#include "esp_random.h"
#include "gate.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"

static const char *TAG = "camlogger";

/* ---- ESP32-S3-EYE pin map (GOOUUU ESP32-S3-CAM) ---- */
#define CAM_PIN_PWDN  -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK  15
#define CAM_PIN_SIOD  4
#define CAM_PIN_SIOC  5
#define CAM_PIN_D7    16
#define CAM_PIN_D6    17
#define CAM_PIN_D5    18
#define CAM_PIN_D4    12
#define CAM_PIN_D3    10
#define CAM_PIN_D2    8
#define CAM_PIN_D1    9
#define CAM_PIN_D0    11
#define CAM_PIN_VSYNC 6
#define CAM_PIN_HREF  7
#define CAM_PIN_PCLK  13

/* thumbnail: SVGA -> 1/8 JPEG decode (100x75) -> 4x4 box -> 25x18 gray */
#define DEC_W 100
#define DEC_H 75
#define TH_W 25
#define TH_H 18

/* state that survives deep sleep */
RTC_DATA_ATTR static uint8_t rtc_thumb[TH_W * TH_H];
RTC_DATA_ATTR static bool rtc_thumb_valid;
RTC_DATA_ATTR static uint32_t rtc_wake_count;
RTC_DATA_ATTR static uint32_t rtc_seq;
/* server-controlled runtime config (fetched after each transmit) */
RTC_DATA_ATTR static uint32_t rtc_cfg_wake_s;
RTC_DATA_ATTR static uint32_t rtc_cfg_heartbeat_n;
RTC_DATA_ATTR static uint32_t rtc_cfg_gate_pct;
RTC_DATA_ATTR static uint8_t rtc_cfg_debug;   /* 1 = transmit every wake */
RTC_DATA_ATTR static uint8_t rtc_cfg_capture; /* 1 = data-collection mode:
                                 transmit every wake; on motion, burst
                                 ~1 fps for 30 s before sleeping */
/* adaptive low-light exposure: 0 = full auto (day), up to EXP_MAX with
   progressively longer manual exposure + gain. Adjusted one step per
   wake from the last frame's thumbnail luma, applied at next cam init. */
#define EXP_MAX 4
RTC_DATA_ATTR static uint8_t rtc_exp_level;
static int s_lum = -1;                 /* this wake's thumbnail mean luma */

static EventGroupHandle_t s_events;
#define WIFI_UP_BIT BIT0

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

static void wifi_start(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
        err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t wcfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wcfg));
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                        &wifi_event, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                        &wifi_event, NULL, NULL);
    wifi_config_t sta = { 0 };
    strlcpy((char *)sta.sta.ssid, CONFIG_CAM_WIFI_SSID,
            sizeof(sta.sta.ssid));
    strlcpy((char *)sta.sta.password, CONFIG_CAM_WIFI_PASS,
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

/* ---------------- camera ---------------- */

static esp_err_t camera_init(void)
{
    camera_config_t cfg = { 0 };
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer = LEDC_TIMER_0;
    cfg.pin_d0 = CAM_PIN_D0;   cfg.pin_d1 = CAM_PIN_D1;
    cfg.pin_d2 = CAM_PIN_D2;   cfg.pin_d3 = CAM_PIN_D3;
    cfg.pin_d4 = CAM_PIN_D4;   cfg.pin_d5 = CAM_PIN_D5;
    cfg.pin_d6 = CAM_PIN_D6;   cfg.pin_d7 = CAM_PIN_D7;
    cfg.pin_xclk = CAM_PIN_XCLK;
    cfg.pin_pclk = CAM_PIN_PCLK;
    cfg.pin_vsync = CAM_PIN_VSYNC;
    cfg.pin_href = CAM_PIN_HREF;
    cfg.pin_sccb_sda = CAM_PIN_SIOD;
    cfg.pin_sccb_scl = CAM_PIN_SIOC;
    cfg.pin_pwdn = CAM_PIN_PWDN;
    cfg.pin_reset = CAM_PIN_RESET;
    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_JPEG;
    cfg.frame_size = FRAMESIZE_SVGA;
    cfg.jpeg_quality = 12;
    cfg.fb_count = 2;
    cfg.fb_location = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) return err;
    /* low-light ladder (OV3660). Levels 2+ go manual: AEC's own ceiling
       is what leaves evening frames dark in the first place. */
    sensor_t *s = esp_camera_sensor_get();
    if (s && rtc_exp_level > 0) {
        if (rtc_exp_level == 1) {
            s->set_ae_level(s, 2);             /* bias auto-exposure up */
            s->set_gainceiling(s, GAINCEILING_32X);
        } else {
            static const int aec[] = { 0, 0, 600, 1000, 1200 };
            static const int agc[] = { 0, 0, 12, 20, 30 };
            s->set_exposure_ctrl(s, 0);
            s->set_aec_value(s, aec[rtc_exp_level]);
            s->set_gain_ctrl(s, 0);
            s->set_agc_gain(s, agc[rtc_exp_level]);
        }
    }
    for (int i = 0; i < 3; i++) {          /* AGC/AWB warm-up */
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        vTaskDelay(pdMS_TO_TICKS(120));
    }
    return ESP_OK;
}

/* JPEG -> 25x18 gray thumbnail. Returns false on decode failure. */
static bool make_thumb(const camera_fb_t *fb, uint8_t *out)
{
    size_t rgb_len = DEC_W * DEC_H * 2;
    uint8_t *rgb = malloc(rgb_len);
    if (!rgb) return false;
    if (!jpg2rgb565(fb->buf, fb->len, rgb, JPG_SCALE_8X)) {
        free(rgb);
        return false;
    }
    for (int ty = 0; ty < TH_H; ty++) {
        for (int tx = 0; tx < TH_W; tx++) {
            int sum = 0;
            for (int dy = 0; dy < 4; dy++) {
                const uint8_t *row = rgb + ((ty * 4 + dy) * DEC_W + tx * 4) * 2;
                for (int dx = 0; dx < 4; dx++) {
                    uint16_t v = (row[dx * 2] << 8) | row[dx * 2 + 1];
                    int r = (v >> 11) & 0x1F, g = (v >> 5) & 0x3F,
                        b = v & 0x1F;
                    sum += (r * 255 / 31 + g * 255 / 63 + b * 255 / 31) / 3;
                }
            }
            out[ty * TH_W + tx] = sum / 16;
        }
    }
    free(rgb);
    return true;
}

/* JPEG -> 96x96x3 int8 gate input: decode 1/4 (200x150), center-crop
   square, nearest-neighbor to 96. Returns false on decode failure. */
static bool make_gate_input(const camera_fb_t *fb, int8_t *out)
{
    const int dw = 200, dh = 150;
    uint8_t *rgb = malloc(dw * dh * 2);
    if (!rgb) return false;
    if (!jpg2rgb565(fb->buf, fb->len, rgb, JPG_SCALE_4X)) {
        free(rgb);
        return false;
    }
    const int side = dh, x0 = (dw - side) / 2;
    for (int y = 0; y < GATE_IMG; y++) {
        int sy = y * side / GATE_IMG;
        for (int x = 0; x < GATE_IMG; x++) {
            int sx = x0 + x * side / GATE_IMG;
            const uint8_t *px = rgb + (sy * dw + sx) * 2;
            uint16_t v = (px[0] << 8) | px[1];
            uint8_t r = ((v >> 11) & 0x1F) * 255 / 31;
            uint8_t g = ((v >> 5) & 0x3F) * 255 / 63;
            uint8_t b = (v & 0x1F) * 255 / 31;
            int8_t *o = out + (y * GATE_IMG + x) * 3;
            o[0] = (int8_t)(r ^ 0x80);
            o[1] = (int8_t)(g ^ 0x80);
            o[2] = (int8_t)(b ^ 0x80);
        }
    }
    free(rgb);
    return true;
}

/* mean|a-b| after mean-luma normalization (exposure-flicker tolerant) */
static int thumb_diff(const uint8_t *a, const uint8_t *b)
{
    int n = TH_W * TH_H, ma = 0, mb = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    int acc = 0;
    for (int i = 0; i < n; i++) {
        int d = (a[i] - ma) - (b[i] - mb);
        acc += d < 0 ? -d : d;
    }
    return acc / n;
}

/* ---------------- upload + ota ---------------- */

static void build_meta(char *meta, size_t sz, int diff, bool heartbeat,
                       int animal_pct, bool audit, const char *why,
                       bool field_tx)
{
    snprintf(meta, sz,
             "{\"seq\":%lu,\"wake\":%lu,\"diff\":%d,\"heartbeat\":%d,"
             "\"animal_pct\":%d,\"audit\":%d,\"debug\":%d,"
             "\"field_tx\":%d,\"why\":\"%s\",\"lum\":%d,\"exp\":%d,"
             "\"fw\":\"%s\"}",
             (unsigned long)rtc_seq, (unsigned long)rtc_wake_count, diff,
             heartbeat ? 1 : 0, animal_pct, audit ? 1 : 0, rtc_cfg_debug,
             field_tx ? 1 : 0, why, s_lum, rtc_exp_level,
             esp_app_get_description()->version);
}

static bool post_jpeg(const uint8_t *buf, size_t len, const char *meta)
{
    char url[160];
    snprintf(url, sizeof(url), "http://%s:%d/blob/%s/frame",
             CONFIG_CAM_SERVER_HOST, CONFIG_CAM_SERVER_PORT,
             CONFIG_CAM_DEVICE_NAME);
    esp_http_client_config_t hc = {
        .url = url, .method = HTTP_METHOD_POST, .timeout_ms = 8000,
    };
    esp_http_client_handle_t c = esp_http_client_init(&hc);
    if (!c) return false;
    esp_http_client_set_header(c, "Content-Type", "image/jpeg");
    esp_http_client_set_header(c, "X-Meta", meta);
    bool ok = false;
    if (esp_http_client_open(c, len) == ESP_OK) {
        if (esp_http_client_write(c, (const char *)buf, len) == (int)len) {
            esp_http_client_fetch_headers(c);
            int st = esp_http_client_get_status_code(c);
            ok = st >= 200 && st < 300;
        }
    }
    esp_http_client_close(c);
    esp_http_client_cleanup(c);
    return ok;
}

static bool post_frame(const camera_fb_t *fb, int diff, bool heartbeat,
                       int animal_pct, bool audit, const char *why,
                       bool field_tx)
{
    char meta[300];
    build_meta(meta, sizeof(meta), diff, heartbeat, animal_pct, audit,
               why, field_tx);
    return post_jpeg(fb->buf, fb->len, meta);
}

/* ---------------- sd card offline buffer ----------------
 * TF slot (back of board): 1-bit SDMMC, CLK=39 CMD=38 D0=40.
 * Card must be FAT32 (<=32GB; format_if_mount_failed is off on purpose).
 * FATFS is built without long-filename support -> strict 8.3 names.
 * When an upload fails, frame + meta go to /sd/pending/NNNNNNNN.jpg/.jsn;
 * after the next successful upload the backlog drains a few per wake. */

#define SD_MNT "/sd"
#define SD_DIR SD_MNT "/pending"
#define SD_FLUSH_MAX 8

static sdmmc_card_t *s_card;
static int s_sd_state;              /* 0 untried, 1 mounted, -1 failed */

static bool sd_mount(void)
{
    if (s_sd_state) return s_sd_state == 1;
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    slot.clk = GPIO_NUM_39;
    slot.cmd = GPIO_NUM_38;
    slot.d0  = GPIO_NUM_40;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    esp_vfs_fat_sdmmc_mount_config_t mc = {
        .format_if_mount_failed = false,
        .max_files = 4,
        .allocation_unit_size = 16 * 1024,
    };
    esp_err_t err = esp_vfs_fat_sdmmc_mount(SD_MNT, &host, &slot, &mc,
                                            &s_card);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "sd: mount failed (%s) - no card?",
                 esp_err_to_name(err));
        s_sd_state = -1;
        return false;
    }
    mkdir(SD_DIR, 0775);
    s_sd_state = 1;
    ESP_LOGI(TAG, "sd: mounted, %llu MB",
             (unsigned long long)s_card->csd.capacity *
             s_card->csd.sector_size / (1024 * 1024));
    return true;
}

static void sd_unmount(void)
{
    if (s_sd_state == 1) {
        esp_vfs_fat_sdcard_unmount(SD_MNT, s_card);
        s_sd_state = 0;
    }
}

/* NVS counter -> filenames unique across power cycles */
static uint32_t sd_next_id(void)
{
    uint32_t v = 0;
    nvs_handle_t h;
    if (nvs_open("camcfg", NVS_READWRITE, &h) != ESP_OK)
        return esp_random();
    nvs_get_u32(h, "sdid", &v);
    v++;
    nvs_set_u32(h, "sdid", v);
    nvs_commit(h);
    nvs_close(h);
    return v;
}

static void sd_save(const camera_fb_t *fb, const char *meta)
{
    if (!sd_mount()) return;
    uint32_t id = sd_next_id();
    char path[48];
    snprintf(path, sizeof(path), SD_DIR "/%08lu.jpg", (unsigned long)id);
    FILE *f = fopen(path, "wb");
    if (!f) { ESP_LOGE(TAG, "sd: open %s failed", path); return; }
    size_t n = fwrite(fb->buf, 1, fb->len, f);
    fclose(f);
    if (n != fb->len) {
        ESP_LOGE(TAG, "sd: short write, dropping");
        unlink(path);
        return;
    }
    snprintf(path, sizeof(path), SD_DIR "/%08lu.jsn", (unsigned long)id);
    f = fopen(path, "w");
    if (f) { fputs(meta, f); fclose(f); }
    ESP_LOGI(TAG, "sd: buffered %08lu.jpg (%u B)",
             (unsigned long)id, (unsigned)fb->len);
}

/* upload + delete up to SD_FLUSH_MAX buffered frames; stop on first
 * failure (server clearly still unhappy - keep the rest for later) */
static void sd_flush(void)
{
    if (!sd_mount()) return;
    DIR *d = opendir(SD_DIR);
    if (!d) return;
    struct dirent *e;
    int sent = 0;
    while (sent < SD_FLUSH_MAX && (e = readdir(d)) != NULL) {
        size_t ln = strlen(e->d_name);
        if (ln < 5 || strcasecmp(e->d_name + ln - 4, ".jpg") != 0)
            continue;
        char jpg[64], jsn[64];
        snprintf(jpg, sizeof(jpg), SD_DIR "/%.13s", e->d_name); /* 8.3 */
        strcpy(jsn, jpg);
        memcpy(jsn + strlen(jsn) - 3, "jsn", 3);

        char meta[320] = "";
        FILE *f = fopen(jsn, "r");
        if (f) {
            size_t n = fread(meta, 1, sizeof(meta) - 20, f);
            meta[n] = 0;
            fclose(f);
        }
        size_t ml = strlen(meta);
        if (ml > 1 && meta[ml - 1] == '}')
            strcpy(meta + ml - 1, ",\"buffered\":1}");
        else
            strcpy(meta, "{\"buffered\":1}");

        struct stat st;
        if (stat(jpg, &st) != 0 || st.st_size <= 0) {
            unlink(jpg); unlink(jsn);
            continue;
        }
        uint8_t *buf = heap_caps_malloc(st.st_size, MALLOC_CAP_SPIRAM);
        if (!buf) break;
        f = fopen(jpg, "rb");
        bool ok = f && fread(buf, 1, st.st_size, f) == (size_t)st.st_size;
        if (f) fclose(f);
        if (ok) ok = post_jpeg(buf, st.st_size, meta);
        free(buf);
        if (!ok) break;
        unlink(jpg);
        unlink(jsn);
        sent++;
    }
    closedir(d);
    if (sent)
        ESP_LOGI(TAG, "sd: flushed %d buffered frame(s)", sent);
}

/* one OTA version check; runs esp_https_ota + restart if server differs */
static void check_ota_once(void)
{
    char url[160], remote[36];
    const char *cur = esp_app_get_description()->version;
    snprintf(url, sizeof(url), "http://%s:%d/firmware/%s/version",
             CONFIG_CAM_SERVER_HOST, CONFIG_CAM_SERVER_PORT,
             CONFIG_CAM_DEVICE_NAME);
    esp_http_client_config_t hc = { .url = url, .timeout_ms = 5000 };
    esp_http_client_handle_t c = esp_http_client_init(&hc);
    int n = 0;
    if (c && esp_http_client_open(c, 0) == ESP_OK) {
        esp_http_client_fetch_headers(c);
        if (esp_http_client_get_status_code(c) == 200) {
            n = esp_http_client_read(c, remote, sizeof(remote) - 1);
        }
    }
    if (c) { esp_http_client_close(c); esp_http_client_cleanup(c); }
    if (n <= 0) return;
    remote[n] = 0;
    while (n > 0 && (remote[n-1] == '\n' || remote[n-1] == '\r'))
        remote[--n] = 0;
    if (!strlen(remote) || strcmp(remote, cur) == 0) return;

    ESP_LOGW(TAG, "OTA %s -> %s", cur, remote);
    snprintf(url, sizeof(url), "http://%s:%d/firmware/%s.bin",
             CONFIG_CAM_SERVER_HOST, CONFIG_CAM_SERVER_PORT,
             CONFIG_CAM_DEVICE_NAME);
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

/* persist config across POWER cycles (RTC only survives deep sleep) */
static void cfg_save_nvs(void)
{
    nvs_handle_t h;
    if (nvs_open("camcfg", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_u32(h, "wake_s", rtc_cfg_wake_s);
    nvs_set_u32(h, "hb_n", rtc_cfg_heartbeat_n);
    nvs_set_u32(h, "gate", rtc_cfg_gate_pct);
    nvs_set_u8(h, "debug", rtc_cfg_debug);
    nvs_set_u8(h, "capt", rtc_cfg_capture);
    nvs_commit(h);
    nvs_close(h);
}

static void cfg_load_nvs(void)
{
    nvs_handle_t h;
    if (nvs_open("camcfg", NVS_READONLY, &h) != ESP_OK) return;
    uint32_t v; uint8_t b;
    if (nvs_get_u32(h, "wake_s", &v) == ESP_OK && v >= 2) rtc_cfg_wake_s = v;
    if (nvs_get_u32(h, "hb_n", &v) == ESP_OK && v >= 1) rtc_cfg_heartbeat_n = v;
    if (nvs_get_u32(h, "gate", &v) == ESP_OK) rtc_cfg_gate_pct = v;
    if (nvs_get_u8(h, "debug", &b) == ESP_OK) rtc_cfg_debug = b;
    if (nvs_get_u8(h, "capt", &b) == ESP_OK) rtc_cfg_capture = b;
    nvs_close(h);
}

/* GET /config/<device>; naive key parsing, missing file keeps current */
static void fetch_config(void)
{
    char url[160], body[256];
    snprintf(url, sizeof(url), "http://%s:%d/config/%s",
             CONFIG_CAM_SERVER_HOST, CONFIG_CAM_SERVER_PORT,
             CONFIG_CAM_DEVICE_NAME);
    esp_http_client_config_t hc = { .url = url, .timeout_ms = 4000 };
    esp_http_client_handle_t c = esp_http_client_init(&hc);
    int n = 0;
    if (c && esp_http_client_open(c, 0) == ESP_OK) {
        esp_http_client_fetch_headers(c);
        if (esp_http_client_get_status_code(c) == 200) {
            n = esp_http_client_read(c, body, sizeof(body) - 1);
        }
    }
    if (c) { esp_http_client_close(c); esp_http_client_cleanup(c); }
    if (n <= 0) return;
    body[n] = 0;
    char *p;
    if ((p = strstr(body, "\"wake_s\""))) {
        int v = atoi(p + 9);
        if (v >= 2 && v <= 3600) rtc_cfg_wake_s = v;
    }
    if ((p = strstr(body, "\"heartbeat_n\""))) {
        int v = atoi(p + 14);
        if (v >= 1 && v <= 10000) rtc_cfg_heartbeat_n = v;
    }
    if ((p = strstr(body, "\"gate_pct\""))) {
        int v = atoi(p + 11);
        if (v >= 0 && v <= 100) rtc_cfg_gate_pct = v;
    }
    rtc_cfg_debug = strstr(body, "\"debug\"") != NULL;
    rtc_cfg_capture = strstr(body, "\"capture\"") != NULL;
    cfg_save_nvs();
    ESP_LOGI(TAG, "config: wake=%lus hb=%lu gate=%lu%% debug=%d capture=%d",
             (unsigned long)rtc_cfg_wake_s,
             (unsigned long)rtc_cfg_heartbeat_n,
             (unsigned long)rtc_cfg_gate_pct, rtc_cfg_debug,
             rtc_cfg_capture);
}

static bool image_pending_verify(void)
{
    esp_ota_img_states_t st;
    const esp_partition_t *run = esp_ota_get_running_partition();
    if (esp_ota_get_state_partition(run, &st) == ESP_OK) {
        return st == ESP_OTA_IMG_PENDING_VERIFY;
    }
    return false;
}

/* ---------------- recovery mode (BOOT held at wake) ---------------- */

static void recovery_loop(void)
{
    ESP_LOGW(TAG, "RECOVERY MODE: continuous frames + OTA polling");
    wifi_start();
    for (;;) {
        if (wifi_wait(15000)) {
            camera_fb_t *fb = esp_camera_fb_get();
            if (fb && fb->len) {
                if (post_frame(fb, -1, false, -1, false, "recovery", true)) {
                    rtc_seq++;
                    if (image_pending_verify())
                        esp_ota_mark_app_valid_cancel_rollback();
                }
            }
            if (fb) esp_camera_fb_return(fb);
            check_ota_once();
        }
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

/* ---------------- main ---------------- */

void app_main(void)
{
    s_events = xEventGroupCreate();
    if (rtc_cfg_wake_s == 0) {          /* power-on: seed defaults */
        rtc_cfg_wake_s = CONFIG_CAM_WAKE_INTERVAL_S;
        rtc_cfg_heartbeat_n = CONFIG_CAM_HEARTBEAT_EVERY_N;
        rtc_cfg_gate_pct = CONFIG_CAM_GATE_THRESHOLD_PCT;
        rtc_cfg_debug = 0;
        nvs_flash_init();               /* idempotent; needed pre-wifi */
        cfg_load_nvs();                 /* survive power cycles */
    }
    rtc_wake_count++;
    bool pending = image_pending_verify();
    bool first_boot = (esp_sleep_get_wakeup_cause()
                       != ESP_SLEEP_WAKEUP_TIMER);

    /* BOOT held -> recovery mode (needs camera too) */
    gpio_config_t io = { .pin_bit_mask = 1ULL << 0,
                         .mode = GPIO_MODE_INPUT,
                         .pull_up_en = GPIO_PULLUP_ENABLE };
    gpio_config(&io);

    esp_err_t cam_ok = camera_init();
    if (gpio_get_level(0) == 0) {
        recovery_loop();               /* never returns */
    }
    if (cam_ok != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed: %s", esp_err_to_name(cam_ok));
        if (pending) esp_restart();    /* trigger rollback */
        esp_deep_sleep(CONFIG_CAM_WAKE_INTERVAL_S * 1000000ULL);
    }

    camera_fb_t *fb = esp_camera_fb_get();
    int diff = -1;
    bool motion = false;
    uint8_t thumb[TH_W * TH_H];
    if (fb && fb->len && make_thumb(fb, thumb)) {
        if (rtc_thumb_valid) {
            diff = thumb_diff(thumb, rtc_thumb);
            motion = diff >= CONFIG_CAM_MOTION_THRESHOLD;
        }
        memcpy(rtc_thumb, thumb, sizeof(rtc_thumb));
        rtc_thumb_valid = true;
        /* exposure feedback for the NEXT wake (sensor re-inits then) */
        int acc = 0;
        for (int i = 0; i < TH_W * TH_H; i++) acc += thumb[i];
        s_lum = acc / (TH_W * TH_H);
        uint8_t prev = rtc_exp_level;
        if (s_lum < 40 && rtc_exp_level < EXP_MAX) rtc_exp_level++;
        else if (s_lum > 110 && rtc_exp_level > 0) rtc_exp_level--;
        if (rtc_exp_level != prev)
            ESP_LOGI(TAG, "lum=%d exp %u -> %u", s_lum, prev, rtc_exp_level);
    }
    bool heartbeat = (rtc_wake_count % rtc_cfg_heartbeat_n) == 0;

    /* animal gate: only consulted for plain motion frames. Fail-open:
       if the model isn't loadable, motion alone transmits (Phase 1). */
    int animal_pct = -1;
    bool audit = false;
    bool gated_motion = motion;
    if (motion && fb && fb->len) {
        static bool gate_ok, gate_tried;
        if (!gate_tried) { gate_tried = true; gate_ok = gate_init(); }
        if (gate_ok) {
            int8_t *gin = heap_caps_malloc(GATE_IMG * GATE_IMG * 3,
                                           MALLOC_CAP_SPIRAM);
            if (gin && make_gate_input(fb, gin)) {
                float s = gate_score(gin);
                if (s >= 0) {
                    animal_pct = (int)(s * 100 + 0.5f);
                    gated_motion = animal_pct >= (int)rtc_cfg_gate_pct;
                    if (!gated_motion &&
                        (esp_random() % CONFIG_CAM_AUDIT_1_IN) == 0) {
                        gated_motion = true;
                        audit = true;
                    }
                }
            }
            if (gin) free(gin);
        }
    }
    /* the verdict deploy mode WOULD give this wake (debug's whole point) */
    const char *why;
    bool field_tx;
    if (heartbeat)            { why = "heartbeat";   field_tx = true; }
    else if (!motion)         { why = "no-motion";   field_tx = false; }
    else if (animal_pct < 0)  { why = "motion";      field_tx = true; }
    else if (audit)           { why = "audit";       field_tx = true; }
    else if (gated_motion)    { why = "animal";      field_tx = true; }
    else                      { why = "gate-reject"; field_tx = false; }

    bool transmit = rtc_cfg_debug || rtc_cfg_capture || gated_motion
                    || heartbeat || first_boot || pending
                    || !rtc_thumb_valid;

    ESP_LOGI(TAG, "wake=%lu diff=%d motion=%d animal=%d%% audit=%d hb=%d "
             "pending=%d -> %s",
             (unsigned long)rtc_wake_count, diff, motion, animal_pct,
             audit, heartbeat, pending, transmit ? "TRANSMIT" : "sleep");

    if (transmit && fb && fb->len) {
        char meta[300];
        build_meta(meta, sizeof(meta), diff, heartbeat, animal_pct, audit,
                   why, field_tx);
        wifi_start();
        if (wifi_wait(12000) && post_jpeg(fb->buf, fb->len, meta)) {
            rtc_seq++;
            if (pending) {
                esp_ota_mark_app_valid_cancel_rollback();
                pending = false;
                ESP_LOGI(TAG, "image marked valid");
            }
            fetch_config();
            check_ota_once();
            sd_flush();                 /* drain offline backlog, if any */
            /* capture mode + motion: stay awake, ~1 fps for 30 s */
            if (rtc_cfg_capture && motion) {
                ESP_LOGI(TAG, "capture burst: 30s @ 1fps");
                int64_t burst_end = esp_timer_get_time() + 30 * 1000000LL;
                while (esp_timer_get_time() < burst_end) {
                    int64_t next = esp_timer_get_time() + 1000000LL;
                    camera_fb_t *bf = esp_camera_fb_get();
                    if (bf && bf->len) {
                        char bm[300];
                        rtc_seq++;
                        build_meta(bm, sizeof(bm), -1, false, -1, false,
                                   "burst", true);
                        post_jpeg(bf->buf, bf->len, bm);
                    }
                    if (bf) esp_camera_fb_return(bf);
                    int64_t wait_us = next - esp_timer_get_time();
                    if (wait_us > 0)
                        vTaskDelay(pdMS_TO_TICKS(wait_us / 1000));
                }
                /* refresh RTC thumbnail from the post-burst scene so
                   next wake's motion diff is against current reality */
                camera_fb_t *bf = esp_camera_fb_get();
                if (bf && bf->len && make_thumb(bf, thumb))
                    memcpy(rtc_thumb, thumb, sizeof(rtc_thumb));
                if (bf) esp_camera_fb_return(bf);
            }
        } else if (pending) {
            /* new image can't reach the server: reboot -> rollback */
            ESP_LOGE(TAG, "pending image failed to upload; rolling back");
            esp_restart();
        } else {
            /* no wifi / server down: buffer to sd card if one is in */
            sd_save(fb, meta);
        }
        esp_wifi_stop();
    }
    if (fb) esp_camera_fb_return(fb);
    esp_camera_deinit();
    sd_unmount();
    esp_deep_sleep((uint64_t)rtc_cfg_wake_s * 1000000ULL);
}
