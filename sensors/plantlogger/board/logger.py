"""FeatherS3 plant data logger -> sensorhub MQTT.

Every INTERVAL_S: read soil moisture/temp (I2C2), ambient light, battery
(MAX17048 on I2C1), append to data.csv on flash, then join Wi-Fi, sync the
clock via NTP, and publish each metric to the home MQTT broker as
home/plant/<device>/<metric> with {"v": value, "ts": unix} (QoS 1).
Unsent readings queue in pending.jsonl and flush on the next connect; the
hub honors the queued ts, so late readings land at measurement time.
On battery power, deep-sleeps between readings; on USB, idles.
"""
import json
import os
import struct
import time

import feathers3
import machine
import network
import ota
import secrets

INTERVAL_S = 3600
RETRY_S = 600  # retry cadence while unsent readings are queued (USB power only)
CSV = "data.csv"
CSV_MAX = 400_000
PENDING = "pending.jsonl"
PENDING_MAX = 200
ERRLOG = "errors.log"

# metrics published to the bus (rec keys; None values are skipped)
METRICS = ("soil_moist", "soil_temp", "amb_light",
           "batt_v", "batt_pct", "batt_rate", "rssi", "vbus")

# MicroPython epoch may be 2000-01-01; unix epoch offset for external timestamps
EPOCH_OFF = 946684800 if time.gmtime(0)[0] == 2000 else 0
DEVICE = "".join("%02x" % b for b in machine.unique_id())

_i2c1 = machine.I2C(0, sda=machine.Pin(8), scl=machine.Pin(9))      # MAX17048 fuel gauge
_i2c2 = machine.I2C(1, sda=machine.Pin(16), scl=machine.Pin(15))    # soil sensor (LDO2)


def log_err(msg):
    try:
        if _size(ERRLOG) > 20_000:
            os.remove(ERRLOG)
        with open(ERRLOG, "a") as f:
            f.write("%d %s\n" % (unix_now(), msg))
    except OSError:
        pass


def _size(path):
    try:
        return os.stat(path)[6]
    except OSError:
        return 0


def unix_now():
    return time.time() + EPOCH_OFF


def clock_synced():
    return time.gmtime()[0] >= 2025


def read_soil():
    _i2c2.writeto(0x36, b"\x0f\x10")
    time.sleep_ms(5)
    r = _i2c2.readfrom(0x36, 2)
    moist = (r[0] << 8) | r[1]
    _i2c2.writeto(0x36, b"\x00\x04")
    time.sleep_ms(5)
    r = _i2c2.readfrom(0x36, 4)
    t = ((r[0] << 24) | (r[1] << 16) | (r[2] << 8) | r[3]) / 65536
    return moist, t


def read_battery():
    def reg(addr, signed=False):
        return struct.unpack(">h" if signed else ">H", _i2c1.readfrom_mem(0x36, addr, 2))[0]
    return reg(0x02) * 78.125e-6, min(reg(0x04) / 256, 100.0), reg(0x16, True) * 0.208


def take_reading():
    feathers3.set_ldo2_power(True)
    time.sleep_ms(300)
    rec = {"device": DEVICE, "uptime_s": time.ticks_ms() // 1000,
           "vbus": feathers3.get_vbus_present(), "rssi": None,
           "ts": unix_now() if clock_synced() else None}
    try:
        rec["soil_moist"], rec["soil_temp"] = read_soil()
    except OSError as e:
        rec["soil_moist"] = rec["soil_temp"] = None
        log_err("soil: %s" % e)
    try:
        rec["amb_light"] = feathers3.get_amb_light()
    except Exception as e:
        rec["amb_light"] = None
        log_err("light: %s" % e)
    try:
        rec["batt_v"], rec["batt_pct"], rec["batt_rate"] = read_battery()
    except OSError as e:
        rec["batt_v"] = rec["batt_pct"] = rec["batt_rate"] = None
        log_err("batt: %s" % e)
    feathers3.set_ldo2_power(False)
    return rec


def append_csv(rec):
    if _size(CSV) > CSV_MAX:
        try:
            os.remove(CSV + ".old")
        except OSError:
            pass
        os.rename(CSV, CSV + ".old")
    new = _size(CSV) == 0
    with open(CSV, "a") as f:
        if new:
            f.write("ts,soil_moist,soil_temp,amb_light,batt_v,batt_pct,batt_rate,vbus,uptime_s,rssi\n")
        f.write("%s,%s,%s,%s,%s,%s,%s,%d,%d,%s\n" % (
            rec["ts"] or "", rec["soil_moist"], rec["soil_temp"], rec["amb_light"],
            rec["batt_v"], rec["batt_pct"], rec["batt_rate"], rec["vbus"], rec["uptime_s"],
            rec["rssi"]))


def queue_pending(rec):
    lines = []
    try:
        with open(PENDING) as f:
            lines = f.readlines()
    except OSError:
        pass
    lines.append(json.dumps(rec) + "\n")
    with open(PENDING, "w") as f:
        for line in lines[-PENDING_MAX:]:
            f.write(line)


def wifi_connect(timeout_s=25):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        try:
            current = wlan.config("ssid")
        except (ValueError, OSError):
            current = None
        if current != secrets.WIFI_SSID:
            wlan.disconnect()
            time.sleep_ms(200)
    if not wlan.isconnected():
        wlan.connect(secrets.WIFI_SSID, secrets.WIFI_PASS)
        t0 = time.ticks_ms()
        while not wlan.isconnected():
            if time.ticks_diff(time.ticks_ms(), t0) > timeout_s * 1000:
                return None
            time.sleep_ms(200)
    return wlan


def ntp_sync(tries=3):
    """Set the RTC from NTP (UTC). Returns True if the clock is usable."""
    import ntptime
    for _ in range(tries):
        try:
            ntptime.settime()
            return True
        except (OSError, OverflowError) as e:
            err = e
            time.sleep_ms(500)
    log_err("ntp: %s" % err)
    return clock_synced()


def mqtt_connect():
    from umqtt.simple import MQTTClient
    c = MQTTClient("plant-" + DEVICE, secrets.MQTT_HOST,
                   user=secrets.MQTT_USER, password=secrets.MQTT_PASS,
                   keepalive=60)
    c.connect()
    try:
        c.sock.settimeout(10)   # bound the QoS1 PUBACK wait
    except AttributeError:
        pass
    return c


def publish_rec(c, rec):
    base = "home/plant/%s/" % rec.get("device", DEVICE)
    for k in METRICS:
        v = rec.get(k)
        if v is None:
            continue
        payload = {"v": int(v) if isinstance(v, bool) else v}
        if rec.get("ts"):
            payload["ts"] = rec["ts"]
        c.publish(base + k, json.dumps(payload), qos=1)


def flush_pending():
    """Publish queued readings; keep whatever fails for the next connect."""
    try:
        with open(PENDING) as f:
            lines = [l for l in f.readlines() if l.strip()]
    except OSError:
        return 0
    sent = 0
    c = None
    try:
        c = mqtt_connect()
        while lines:
            publish_rec(c, json.loads(lines[0]))
            lines.pop(0)
            sent += 1
        if sent:
            # the queue is fully PUBACK'd: any pending update just proved
            # itself -- mark it valid, then report the (updated) version
            ota.mark_valid()
            c.publish("home/plant/%s/fw" % DEVICE,
                      json.dumps({"ver": ota._read(ota.VERSION_FILE) or "usb"}))
    except (OSError, ValueError) as e:
        log_err("mqtt: %s" % e)
    finally:
        if c:
            try:
                c.disconnect()
            except OSError:
                pass
    if lines:
        with open(PENDING, "w") as f:
            for line in lines:
                f.write(line)
    else:
        try:
            os.remove(PENDING)
        except OSError:
            pass
    return sent


def cycle():
    feathers3.led_set(True)
    rec = take_reading()
    wlan = wifi_connect()
    if wlan:
        try:
            rec["rssi"] = wlan.status("rssi")
        except (ValueError, OSError):
            pass
        if ntp_sync() and rec["ts"] is None:
            rec["ts"] = unix_now() - 30   # reading was taken just before sync
    queue_pending(rec)
    sent = 0
    if wlan:
        sent = flush_pending()   # marks a pending update valid on full flush
        ota.check(rec["batt_pct"], rec["vbus"], log_err)  # may reset
    append_csv(rec)
    if wlan:
        wlan.active(False)
    feathers3.led_set(False)
    return rec, sent


def main():
    while True:
        t0 = time.ticks_ms()
        try:
            cycle()
        except Exception as e:
            log_err("cycle: %r" % e)
        interval_s = INTERVAL_S
        try:
            os.stat(PENDING)  # unsent readings queued
            if feathers3.get_vbus_present():
                interval_s = RETRY_S
        except OSError:
            pass
        elapsed_ms = time.ticks_diff(time.ticks_ms(), t0)
        remain_ms = max(interval_s * 1000 - elapsed_ms, 60_000)
        if not feathers3.get_vbus_present():
            machine.deepsleep(remain_ms)  # wakes by rebooting into main.py
        time.sleep_ms(remain_ms)
