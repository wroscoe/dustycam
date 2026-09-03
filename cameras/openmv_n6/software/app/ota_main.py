"""OTA bootstrap. Deployed as /flash/main.py — tiny and NEVER updated OTA;
only app.py changes over the air, so a bad push can't brick the update loop.

Boot: WiFi -> NTP -> start OTA listener -> run app.run(ota.poll).
On app crash: restore app_prev.py (keeping the bad file as app_bad.py) and
reboot; if there is nothing to restore, sit in recovery mode polling OTA
so a fixed app.py can be pushed.
"""
import os
import sys
import time

import machine

import ota
import secrets

wlan = ota.wifi_connect(secrets.WIFI_SSID, secrets.WIFI_PASS)
print('wifi:', wlan.ifconfig()[0] if wlan else 'FAILED')
try:
    import ntptime
    ntptime.settime()
    print('ntp: synced')
except Exception:
    print('ntp: skipped')

ota.start(getattr(secrets, 'OTA_PORT', 8266), getattr(secrets, 'OTA_TOKEN', ''))

try:
    import app
    ota.state['version'] = getattr(app, 'APP_VERSION', '?')
    print('app version:', ota.state['version'])
    app.run(ota.poll)
except Exception as e:
    sys.print_exception(e)
    try:
        os.stat('/flash/app_prev.py')       # anything to roll back to?
        try:
            os.remove('/flash/app_bad.py')
        except OSError:
            pass
        os.rename('/flash/app.py', '/flash/app_bad.py')
        os.rename('/flash/app_prev.py', '/flash/app.py')
        print('ota: app crashed, reverted to previous version, rebooting')
        time.sleep(1)
        machine.reset()
    except OSError:
        pass                                # no fallback available

print('ota: RECOVERY MODE — push a fixed app.py to /update')
ota.state['version'] = 'recovery'
while True:
    ota.poll()
    time.sleep_ms(100)
