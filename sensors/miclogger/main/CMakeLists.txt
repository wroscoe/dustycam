idf_component_register(
    SRCS "main.c"
    PRIV_REQUIRES esp_wifi esp_http_client nvs_flash esp_netif
                  esp_https_ota app_update esp_timer esp_driver_i2s
    INCLUDE_DIRS "")
