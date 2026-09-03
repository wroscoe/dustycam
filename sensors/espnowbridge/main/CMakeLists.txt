idf_component_register(
    SRCS "main.c"
    PRIV_REQUIRES esp_wifi esp_netif nvs_flash esp_timer esp_http_client
                  esp_https_ota app_update mqtt
    INCLUDE_DIRS "")
