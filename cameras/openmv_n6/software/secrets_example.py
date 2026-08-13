# Copy to secrets.py and fill in, then deploy to /flash/secrets.py on the N6.
# secrets.py is gitignored and never committed. Keep the master copy in
# ~/.dusty/ and copy values from there so all cameras stay consistent.
#
# NOT OTA-managed: after editing, `mpremote cp secrets.py :/flash` and reset,
# or the app runs stale config. New knobs should be read with getattr defaults
# so an older secrets.py keeps working.
WIFI_SSID = 'your-ssid'
WIFI_PASS = 'your-password'

SERVER_HOST = '192.168.1.100'    # sensorhub ingest host
SERVER_PORT = 8088
DEVICE = 'n6cam'

MQTT_USER = ''
MQTT_PASS = ''
MQTT_TOPIC = 'home/cam/n6cam'

OTA_PORT = 8266
OTA_TOKEN = 'a-long-random-hex-string'   # shared secret for ./ota_push.py

PERIOD_S = 10                    # seconds between change checks
DIFF_MIN_FRAC = 0.005            # upload if >0.5% of pixels changed
DIFF_L_THRESH = 8                # per-pixel lightness delta (of 100) counted
HEARTBEAT_S = 300                # force an upload at least this often
TELEMETRY_S = 60
