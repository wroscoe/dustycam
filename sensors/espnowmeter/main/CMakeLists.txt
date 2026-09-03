idf_component_register(
    SRCS "main.c"
    PRIV_REQUIRES esp_wifi esp_netif nvs_flash esp_timer esp_driver_gpio
    INCLUDE_DIRS "")
