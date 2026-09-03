/* WiFi + HTTP uploader for (frame, score) samples.
 *
 * Samples queue in PSRAM (survives offline periods; oldest dropped when
 * full). A background task drains the queue whenever WiFi is up. Wifi
 * credentials/server come from Kconfig (sdkconfig.secrets, not committed).
 */
#include "uploader.h"

#include <string.h>
#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "esp_event.h"
#include "esp_http_client.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "esp_https_ota.h"
#include "esp_ota_ops.h"
#include "esp_app_desc.h"

#define IMG_W 96
#define IMG_H 96
#define IMG_BYTES (IMG_W * IMG_H)
#define QUEUE_DEPTH 64          /* 64 * ~9.3KB in PSRAM */

static const char *TAG = "uploader";

typedef struct {
    uint8_t img[IMG_BYTES];
    float person;
    float no_person;
    uint32_t seq;
} sample_t;

static QueueHandle_t s_queue;
static EventGroupHandle_t s_events;
static uint32_t s_seq;
static char s_device[13];
#define WIFI_UP_BIT BIT0

#ifndef CONFIG_SAMPLE_WIFI_SSID
#define CONFIG_SAMPLE_WIFI_SSID ""
#define CONFIG_SAMPLE_WIFI_PASS ""
#define CONFIG_SAMPLE_SERVER_HOST ""
#define CONFIG_SAMPLE_SERVER_PORT 8077
#endif

static void wifi_event(void *arg, esp_event_base_t base, int32_t id,
                       void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_events, WIFI_UP_BIT);
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(TAG, "wifi connected");
        xEventGroupSetBits(s_events, WIFI_UP_BIT);
    }
}

static bool post_sample(const sample_t *s)
{
    char url[128], hdr[32];
    snprintf(url, sizeof(url), "http://%s:%d/sample",
             CONFIG_SAMPLE_SERVER_HOST, CONFIG_SAMPLE_SERVER_PORT);
    esp_http_client_config_t cfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 5000,
    };
    esp_http_client_handle_t c = esp_http_client_init(&cfg);
    if (!c) return false;
    esp_http_client_set_header(c, "Content-Type", "application/octet-stream");
    snprintf(hdr, sizeof(hdr), "%lu", (unsigned long)s->seq);
    esp_http_client_set_header(c, "X-Seq", hdr);
    snprintf(hdr, sizeof(hdr), "%.4f", s->person);
    esp_http_client_set_header(c, "X-Person-Score", hdr);
    snprintf(hdr, sizeof(hdr), "%.4f", s->no_person);
    esp_http_client_set_header(c, "X-No-Person-Score", hdr);
    snprintf(hdr, sizeof(hdr), "%dx%d", IMG_W, IMG_H);
    esp_http_client_set_header(c, "X-Size", hdr);
    esp_http_client_set_header(c, "X-Device", s_device);
    bool ok = false;
    if (esp_http_client_open(c, IMG_BYTES) == ESP_OK) {
        if (esp_http_client_write(c, (const char *)s->img, IMG_BYTES)
            == IMG_BYTES) {
            esp_http_client_fetch_headers(c);
            int status = esp_http_client_get_status_code(c);
            ok = status >= 200 && status < 300;
        }
    }
    esp_http_client_close(c);
    esp_http_client_cleanup(c);
    return ok;
}

static void ota_task(void *arg)
{
    char url[128], cur[32], remote[36];
    const esp_app_desc_t *desc = esp_app_get_description();
    strlcpy(cur, desc->version, sizeof(cur));
    ESP_LOGI(TAG, "running version %s", cur);
    for (;;) {
        xEventGroupWaitBits(s_events, WIFI_UP_BIT, pdFALSE, pdTRUE,
                            portMAX_DELAY);
        snprintf(url, sizeof(url), "http://%s:%d/firmware/version",
                 CONFIG_SAMPLE_SERVER_HOST, CONFIG_SAMPLE_SERVER_PORT);
        esp_http_client_config_t cfg = {
            .url = url, .timeout_ms = 5000,
        };
        esp_http_client_handle_t c = esp_http_client_init(&cfg);
        int n = 0;
        if (c && esp_http_client_open(c, 0) == ESP_OK) {
            esp_http_client_fetch_headers(c);
            if (esp_http_client_get_status_code(c) == 200) {
                n = esp_http_client_read(c, remote, sizeof(remote) - 1);
            }
        }
        if (c) { esp_http_client_close(c); esp_http_client_cleanup(c); }
        if (n > 0) {
            remote[n] = 0;
            /* trim whitespace */
            while (n > 0 && (remote[n-1] == '\n' || remote[n-1] == '\r'
                             || remote[n-1] == ' ')) remote[--n] = 0;
            if (strlen(remote) > 0 && strcmp(remote, cur) != 0) {
                ESP_LOGW(TAG, "OTA: %s -> %s, updating...", cur, remote);
                snprintf(url, sizeof(url), "http://%s:%d/firmware.bin",
                         CONFIG_SAMPLE_SERVER_HOST,
                         CONFIG_SAMPLE_SERVER_PORT);
                esp_http_client_config_t ota_http = {
                    .url = url, .timeout_ms = 15000, .keep_alive_enable = true,
                };
                esp_https_ota_config_t ota_cfg = {
                    .http_config = &ota_http,
                };
                esp_err_t err = esp_https_ota(&ota_cfg);
                if (err == ESP_OK) {
                    ESP_LOGW(TAG, "OTA ok, rebooting");
                    vTaskDelay(pdMS_TO_TICKS(500));
                    esp_restart();
                }
                ESP_LOGE(TAG, "OTA failed: %s", esp_err_to_name(err));
            }
        }
        vTaskDelay(pdMS_TO_TICKS(30000));
    }
}

static void upload_task(void *arg)
{
    sample_t *s = NULL;
    for (;;) {
        xEventGroupWaitBits(s_events, WIFI_UP_BIT, pdFALSE, pdTRUE,
                            portMAX_DELAY);
        if (xQueueReceive(s_queue, &s, pdMS_TO_TICKS(1000)) != pdTRUE) {
            continue;
        }
        if (post_sample(s)) {
            static bool s_marked;
            if (!s_marked) {
                /* camera + wifi + server all proven: this image is good */
                esp_ota_mark_app_valid_cancel_rollback();
                s_marked = true;
            }
            free(s);
        } else {
            /* server unreachable: put it back and breathe */
            xQueueSendToFront(s_queue, &s, 0);
            vTaskDelay(pdMS_TO_TICKS(3000));
        }
    }
}

void uploader_init(void)
{
    if (strlen(CONFIG_SAMPLE_WIFI_SSID) == 0) {
        ESP_LOGW(TAG, "no wifi ssid configured; uploads disabled");
        return;
    }
    s_events = xEventGroupCreate();
    s_queue = xQueueCreate(QUEUE_DEPTH, sizeof(sample_t *));

    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(s_device, sizeof(s_device), "%02x%02x%02x%02x%02x%02x",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

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
    strlcpy((char *)sta.sta.ssid, CONFIG_SAMPLE_WIFI_SSID,
            sizeof(sta.sta.ssid));
    strlcpy((char *)sta.sta.password, CONFIG_SAMPLE_WIFI_PASS,
            sizeof(sta.sta.password));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
    ESP_ERROR_CHECK(esp_wifi_start());

    xTaskCreate(upload_task, "uploader", 4096, NULL, 4, NULL);
    xTaskCreate(ota_task, "ota", 8192, NULL, 3, NULL);
    ESP_LOGI(TAG, "uploader ready (server %s:%d)",
             CONFIG_SAMPLE_SERVER_HOST, CONFIG_SAMPLE_SERVER_PORT);
}

void uploader_submit(const int8_t *img, float person_score,
                     float no_person_score)
{
    if (!s_queue) return;
    sample_t *s = heap_caps_malloc(sizeof(sample_t),
                                   MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s) return;
    for (int i = 0; i < IMG_BYTES; i++) {
        s->img[i] = (uint8_t)(img[i] ^ 0x80);   /* de-quantize int8->uint8 */
    }
    s->person = person_score;
    s->no_person = no_person_score;
    s->seq = s_seq++;
    if (xQueueSend(s_queue, &s, 0) != pdTRUE) {
        /* full: drop the OLDEST to keep the freshest data */
        sample_t *old;
        if (xQueueReceive(s_queue, &old, 0) == pdTRUE) free(old);
        if (xQueueSend(s_queue, &s, 0) != pdTRUE) free(s);
    }
}
