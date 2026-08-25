"""Wi-Fi scan: list visible access points sorted by signal strength."""
import network

AUTH = {
    0: "open",
    1: "WEP",
    2: "WPA-PSK",
    3: "WPA2-PSK",
    4: "WPA/WPA2",
    5: "WPA2-ENT",
    6: "WPA3-PSK",
    7: "WPA2/WPA3",
}

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
nets = wlan.scan()

print("found %d networks:" % len(nets))
for ssid, bssid, channel, rssi, security, hidden in sorted(nets, key=lambda n: -n[3]):
    mac = ":".join("%02X" % b for b in bssid)
    name = ssid.decode() if ssid else "<hidden>"
    print(
        "%-32s  %s  ch%-2d  %4d dBm  %s"
        % (name, mac, channel, rssi, AUTH.get(security, str(security)))
    )

wlan.active(False)
