/* espnowmeter — Heltec ESP32-S3: roving ESP-NOW signal-strength meter.
 *
 *   Promiscuous sniffer parked on the fleet's ESP-NOW channel. Every
 *   ESP-NOW frame in the air whose payload starts "MX" (the miclogger's
 *   audio, unicast or not) is an RSSI sample. Samples aggregate into 5 s
 *   windows {frames, rssi avg/max} kept in a ring, and windows are sent
 *   to the espnowbridge as unicast "SS" frames — unicast so the MAC ACK
 *   confirms delivery; unACKed windows stay queued and flush when the
 *   meter walks back into bridge range. Carry it into the dead zone and
 *   the survey comes home with you.
 *
 *   LED: short blink per sniffed MX frame (live "the mic reaches here"),
 *   solid 200 ms every send that the bridge ACKs ("I reach the bridge").
 *
 *   SS payload: "SS" | ver u8 | count u8 | count x { age_s u16,
 *   frames u16, rssi_avg i8, rssi_max i8 }.  rssi -127 = no frames.
 */
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "driver/gpio.h"

static const char *TAG = "meter";

#define CHANNEL    CONFIG_METER_CHANNEL
#define LED_GPIO   CONFIG_METER_LED_GPIO
#define WINDOW_S   5
#define RING       256              /* ~21 min of survey backlog */
#define PER_SEND   4                /* windows per SS frame */

typedef struct {
    int64_t t_us;                   /* window close time */
    uint16_t frames;
    int16_t rssi_sum;
    int8_t rssi_max;
    bool sent;
} win_t;

static win_t s_ring[RING];
static int s_head;                  /* next slot to write */
static volatile uint32_t s_win_frames;
static volatile int32_t s_win_rssi_sum;
static volatile int8_t s_win_rssi_max = -127;
static volatile int64_t s_led_until;

static uint8_t s_bridge_mac[6];
static SemaphoreHandle_t s_sent;
static volatile bool s_acked;

/* ---------------- sniffer ---------------- */

static void promis_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    if (type != WIFI_PKT_MGMT)
        return;
    const wifi_promiscuous_pkt_t *p = buf;
    int len = p->rx_ctrl.sig_len;
    if (len < 40 || len > 1500)
        return;
    const uint8_t *d = p->payload;
    /* ESP-NOW = action frame carrying vendor IE dd ?? 18 fe 34 04 */
    int lim = len - 16;
    for (int i = 24; i < lim && i < 80; i++) {
        if (d[i] == 0xdd && d[i + 2] == 0x18 && d[i + 3] == 0xfe
            && d[i + 4] == 0x34 && d[i + 5] == 0x04) {
            const uint8_t *pl = d + i + 7;
            if (pl[0] == 'M' && pl[1] == 'X') {
                s_win_frames++;
                s_win_rssi_sum += p->rx_ctrl.rssi;
                if (p->rx_ctrl.rssi > s_win_rssi_max)
                    s_win_rssi_max = p->rx_ctrl.rssi;
                s_led_until = esp_timer_get_time() + 30000;
            }
            return;
        }
    }
}

/* ---------------- reporting ---------------- */

static void send_cb(const esp_now_send_info_t *info,
                    esp_now_send_status_t status)
{
    s_acked = status == ESP_NOW_SEND_SUCCESS;
    xSemaphoreGive(s_sent);
}

static void close_window(void)
{
    win_t *w = &s_ring[s_head];
    w->t_us = esp_timer_get_time();
    w->frames = s_win_frames;
    w->rssi_sum = s_win_frames ? s_win_rssi_sum : 0;
    w->rssi_max = s_win_frames ? s_win_rssi_max : -127;
    w->sent = false;
    s_head = (s_head + 1) % RING;
    s_win_frames = 0;
    s_win_rssi_sum = 0;
    s_win_rssi_max = -127;
}

static int fill_ss(uint8_t *buf)
{
    int cnt = 0;
    int64_t now = esp_timer_get_time();
    buf[0] = 'S'; buf[1] = 'S'; buf[2] = 1;
    uint8_t *w = buf + 4;
    /* newest-first so live readings win; older backlog rides along */
    for (int k = 1; k <= RING && cnt < PER_SEND; k++) {
        int i = (s_head - k + RING) % RING;
        if (s_ring[i].t_us == 0 || s_ring[i].sent)
            continue;
        uint16_t age = (now - s_ring[i].t_us) / 1000000;
        uint16_t fr = s_ring[i].frames;
        int8_t avg = fr ? s_ring[i].rssi_sum / (int)fr : -127;
        memcpy(w, &age, 2);
        memcpy(w + 2, &fr, 2);
        w[4] = (uint8_t)avg;
        w[5] = (uint8_t)s_ring[i].rssi_max;
        w += 6;
        cnt++;
    }
    buf[3] = cnt;
    return cnt ? 4 + cnt * 6 : 0;
}

static void mark_sent(int cnt)
{
    for (int k = 1; k <= RING && cnt > 0; k++) {
        int i = (s_head - k + RING) % RING;
        if (s_ring[i].t_us == 0 || s_ring[i].sent)
            continue;
        s_ring[i].sent = true;
        cnt--;
    }
}

static void meter_task(void *arg)
{
    uint8_t buf[4 + PER_SEND * 6];
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(WINDOW_S * 1000));
        close_window();
        int len = fill_ss(buf);
        if (!len)
            continue;
        s_acked = false;
        if (esp_now_send(s_bridge_mac, buf, len) == ESP_OK)
            xSemaphoreTake(s_sent, pdMS_TO_TICKS(200));
        if (s_acked) {
            mark_sent(buf[3]);
            gpio_set_level(LED_GPIO, 1);        /* bridge in range */
            vTaskDelay(pdMS_TO_TICKS(200));
            gpio_set_level(LED_GPIO, 0);
        }
    }
}

static void led_task(void *arg)
{
    for (;;) {
        gpio_set_level(LED_GPIO,
                       esp_timer_get_time() < s_led_until ? 1 : 0);
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t wcfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wcfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    /* LR bit so the sniffer decodes the miclogger's long-range frames */
    ESP_ERROR_CHECK(esp_wifi_set_protocol(
        WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G |
        WIFI_PROTOCOL_11N | WIFI_PROTOCOL_LR));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE));

    unsigned m[6];
    if (sscanf(CONFIG_METER_BRIDGE_MAC, "%x:%x:%x:%x:%x:%x",
               &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]) == 6)
        for (int i = 0; i < 6; i++)
            s_bridge_mac[i] = m[i];

    ESP_ERROR_CHECK(esp_now_init());
    s_sent = xSemaphoreCreateBinary();
    esp_now_register_send_cb(send_cb);
    esp_now_peer_info_t peer = { 0 };
    memcpy(peer.peer_addr, s_bridge_mac, 6);
    peer.channel = CHANNEL;
    peer.ifidx = WIFI_IF_STA;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    wifi_promiscuous_filter_t filt = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT,
    };
    esp_wifi_set_promiscuous_filter(&filt);
    esp_wifi_set_promiscuous_rx_cb(promis_cb);
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    gpio_config_t io = {
        .pin_bit_mask = 1ULL << LED_GPIO,
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io);

    ESP_LOGI(TAG, "meter up: sniffing ch %d, bridge %s",
             CHANNEL, CONFIG_METER_BRIDGE_MAC);
    xTaskCreate(led_task, "led", 2048, NULL, 5, NULL);
    xTaskCreate(meter_task, "meter", 4096, NULL, 10, NULL);
}
