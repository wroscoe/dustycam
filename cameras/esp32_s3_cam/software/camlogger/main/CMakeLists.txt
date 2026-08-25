idf_component_register(
    SRCS "main.c" "gate.cc" "gate_model_data.cc"
    PRIV_REQUIRES esp_wifi esp_http_client nvs_flash esp_netif
                  esp_https_ota app_update esp_timer
                  fatfs sdmmc esp_driver_sdmmc
    INCLUDE_DIRS "")
