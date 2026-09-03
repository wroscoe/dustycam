# Do not hand-edit secrets.py for this camera — run ./gen_secrets.py [--public],
# which fills it from ~/.dusty/config.toml + secrets.toml. Keys it writes:
#   WIFI_SSID WIFI_PASS SERVER_HOST SERVER_PORT SERVER_TLS BLOB_TOKEN DEVICE
#   MQTT_USER MQTT_PASS MQTT_TOPIC OTA_PORT OTA_TOKEN PERIOD_S DIFF_MIN_FRAC
#   DIFF_L_THRESH HEARTBEAT_S TELEMETRY_S
# Optional knobs read with getattr defaults: JPEG_QUALITY (85), MAX_PENDING
# (2000), WIFI_RETRY_S (30), TELEMETRY_MQTT (False).
