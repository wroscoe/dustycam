
#
# Main component of TF Micro project 'person_detection'.
#

idf_component_register(
    SRCS
        "detection_responder.cc"
        "image_provider.cc"
        "main.cc"
        "main_functions.cc"
        "model_settings.cc"
        "person_detect_model_data.cc"
        "app_camera_esp.c"
        "esp_cli.c"
        "uploader.c"

    PRIV_REQUIRES console static_images spi_flash esp-tflite-micro esp_wifi esp_http_client nvs_flash esp_netif esp_https_ota app_update
    INCLUDE_DIRS "")
