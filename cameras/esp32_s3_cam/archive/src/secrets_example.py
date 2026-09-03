# Copy to src/secrets.py and fill in, then deploy it to the board.
# secrets.py is gitignored and never committed. Keep the master copy in
# ~/.dusty/ and copy values from there so all cameras stay consistent.
WIFI_SSID = 'your-ssid'
WIFI_PASS = 'your-password'

SERVER_HOST = '192.168.1.100'    # this workstation's LAN IP
SERVER_PORT = 8077

# boot_cam.py starts WebREPL only when this is set — an unauthenticated
# WebREPL would expose a REPL on the LAN.
WEBREPL_PASS = 'choose-one'
