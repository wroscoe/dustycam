# boot.py — Waveshare ESP32-S3-CAM: join Wi-Fi and start WebREPL so scripts
# can be pushed over the network (USB on this board is temperamental).
import network
import time

# Credentials live in secrets.py (deployed to the board, never committed) —
# same convention as src/uplink.py. Copy src/secrets_example.py to make one.
try:
    import secrets
except ImportError:
    secrets = None

WIFI_SSID = secrets.WIFI_SSID if secrets else None
WIFI_PASS = secrets.WIFI_PASS if secrets else None
WEBREPL_PASS = getattr(secrets, "WEBREPL_PASS", None) if secrets else None

try:
    network.hostname("wavecam")
except Exception:
    pass

sta = network.WLAN(network.STA_IF)
sta.active(True)
if not sta.isconnected():
    sta.connect(WIFI_SSID, WIFI_PASS)
    t0 = time.ticks_ms()
    while not sta.isconnected() and time.ticks_diff(time.ticks_ms(), t0) < 15000:
        time.sleep_ms(200)

if sta.isconnected():
    print("wifi:", sta.ifconfig()[0])
    if WEBREPL_PASS:
        import webrepl
        webrepl.start(password=WEBREPL_PASS)
    else:
        # No password in secrets.py — starting WebREPL open would expose a
        # REPL on the LAN, so stay off and say why.
        print("webrepl: no WEBREPL_PASS in secrets.py, not starting")
else:
    print("wifi: connect failed")
