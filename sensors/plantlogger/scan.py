"""BLE scan: listen for advertisements for 8 s and print unique devices by RSSI."""
import bluetooth
import time

_IRQ_SCAN_RESULT = 5
_IRQ_SCAN_DONE = 6

devices = {}  # addr bytes -> [addr_type, best_rssi, name, count]
done = False


def decode_name(adv_data):
    # AD structures: [len][type][payload...]; name types 0x08 (short) / 0x09 (full)
    i = 0
    data = bytes(adv_data)
    while i + 1 < len(data):
        length = data[i]
        if length == 0:
            break
        ad_type = data[i + 1]
        if ad_type in (0x08, 0x09):
            try:
                return data[i + 2 : i + 1 + length].decode()
            except UnicodeError:
                return None
        i += 1 + length
    return None


def irq(event, data):
    global done
    if event == _IRQ_SCAN_RESULT:
        addr_type, addr, adv_type, rssi, adv_data = data
        key = bytes(addr)
        entry = devices.get(key)
        name = decode_name(adv_data)
        if entry is None:
            devices[key] = [addr_type, rssi, name, 1]
        else:
            entry[1] = max(entry[1], rssi)
            entry[3] += 1
            if name and not entry[2]:
                entry[2] = name
    elif event == _IRQ_SCAN_DONE:
        done = True


ble = bluetooth.BLE()
ble.active(True)
ble.irq(irq)
print("scanning 8s...")
# interval/window us: listen continuously, active scan (request scan responses for names)
ble.gap_scan(8000, 30000, 30000, True)

while not done:
    time.sleep_ms(100)

print("found %d devices:" % len(devices))
for key in sorted(devices, key=lambda k: -devices[k][1]):
    addr_type, rssi, name, count = devices[key]
    mac = ":".join("%02X" % b for b in key)
    kind = "public" if addr_type == 0 else "random"
    print("%s  %4d dBm  %-6s  x%-3d  %s" % (mac, rssi, kind, count, name or "-"))

ble.active(False)
