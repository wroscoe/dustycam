"""N6 -> sensorhub uploader with motion gating, MQTT telemetry, and SD
offline buffering. Deployed as /flash/app.py, run by the OTA bootstrap
(ota_main.py as /flash/main.py) — update over WiFi with ./ota_push.py.

Normal operation: every PERIOD_S seconds, diff the frame against the last
*recorded* frame and POST a VGA JPEG to the sensorhub ingest when enough
pixels changed or HEARTBEAT_S passed. Telemetry (battery, RSSI, counters,
pending backlog) publishes over MQTT every TELEMETRY_S.

Offline: when WiFi or the server is unreachable, motion/heartbeat frames
are written to /sdcard/pending/ (JPEG + .json meta sidecar with the real
capture ts, tagged "buffered": true). WiFi rejoin is attempted every 30 s.
When an upload succeeds again, recording pauses and the entire backlog
drains oldest-first (ota.poll() runs between files so pushes still land),
then live operation resumes. No SD card -> buffering quietly disabled.

Hard-won constraints baked in (details in sargineer):
- No pyb.ADCAll/read_core_temp: hard-hangs the MCU while the CSI streams
  (USB gone, power LED on, physical replug required; bypasses rollback).
- Heap image.Image() buffers, not alloc_extra_fb (deprecated in fw 5.0).
- Raw-socket POST: frozen `requests` chokes on the ingest's HTTP/1.0 reply.
- secrets.py is NOT OTA-managed; new config knobs use getattr defaults.
"""
import gc
import math
import os
import socket
import time

import image
import imu
import machine
import mqtt
import network
import sensor

import ota
import secrets

APP_VERSION = '1.7'

# Fast battery watch: publish batt_v/charging every N seconds from the idle
# loop (0 = off). Hardcoded, not in secrets — secrets isn't OTA-managed.
# Flip to 2 and OTA-push when live battery monitoring is needed.
BATT_WATCH_S = 0

# Fast speed watch: publish speed_mps every N seconds (0 = off). On for the
# IMU-speed demo; costs one small MQTT publish per interval.
SPEED_WATCH_S = 2

EPOCH_OFFSET = 946684800 if time.time() < 1_000_000_000 else 0  # 2000 vs 1970 epoch

DIFF_MIN_FRAC = getattr(secrets, 'DIFF_MIN_FRAC', 0.005)
DIFF_L_THRESH = getattr(secrets, 'DIFF_L_THRESH', 8)
HEARTBEAT_S = getattr(secrets, 'HEARTBEAT_S', 300)
TELEMETRY_S = getattr(secrets, 'TELEMETRY_S', 60)
MAX_PENDING = getattr(secrets, 'MAX_PENDING', 2000)
WIFI_RETRY_S = getattr(secrets, 'WIFI_RETRY_S', 30)

PENDING_DIR = '/sdcard/pending'
BATT_DIVIDER = 1.5   # BAT_ADC divider ratio (inferred; verify against a real pack)

_bat_adc = machine.ADC(machine.Pin.board.BAT_ADC)
_chg_pin = machine.Pin(machine.Pin.board.CHG, machine.Pin.IN, machine.Pin.PULL_UP)
_mq = None


def batt_volts():
    raw = sum(_bat_adc.read_u16() for _ in range(8)) // 8
    return raw / 65535 * 3.3 * BATT_DIVIDER


# --- IMU speed estimate (dead reckoning + ZUPT) -----------------------------
# Integrating accelerometer data drifts without bound, so this only stays
# honest through zero-velocity updates: whenever the IMU looks still for
# ~0.5 s (linear accel and gyro both quiet), velocity clamps to zero. Good
# for speed during motion bursts (carried/waved/driven); NOT a speedometer
# for long constant-velocity travel, which an IMU alone cannot observe.

GRAV_ALPHA = 0.02        # gravity low-pass rate per sample
STILL_ACC_MG = 60        # linear-accel magnitude regarded as "still"
STILL_GYRO_MDPS = 3000   # gyro magnitude regarded as "still"
STILL_ZUPT_N = 25        # consecutive still samples (~0.5 s at 50 Hz) -> v=0
VEL_LEAK = 0.998         # per-sample decay bounding residual drift

_grav = None
_vel = [0.0, 0.0, 0.0]
_still_n = 0
_speed = 0.0
_speed_max = 0.0
_last_imu_ms = None


def imu_speed_sample():
    """One IMU sample -> updated speed estimate (m/s)."""
    global _grav, _still_n, _speed, _speed_max, _last_imu_ms
    now = time.ticks_ms()
    dt = 0.02 if _last_imu_ms is None else \
        min(time.ticks_diff(now, _last_imu_ms), 200) / 1000
    _last_imu_ms = now

    ax, ay, az = imu.acceleration_mg()
    gx, gy, gz = imu.angular_rate_mdps()
    if _grav is None:
        _grav = [ax, ay, az]
    lin = [0.0, 0.0, 0.0]
    for i, a in enumerate((ax, ay, az)):
        _grav[i] += GRAV_ALPHA * (a - _grav[i])
        lin[i] = a - _grav[i]
    la = math.sqrt(lin[0] ** 2 + lin[1] ** 2 + lin[2] ** 2)
    gmag = math.sqrt(gx * gx + gy * gy + gz * gz)

    if la < STILL_ACC_MG and gmag < STILL_GYRO_MDPS:
        _still_n += 1
    else:
        _still_n = 0
    if _still_n >= STILL_ZUPT_N:
        _vel[0] = _vel[1] = _vel[2] = 0.0
        _speed = 0.0
        return 0.0

    for i in range(3):
        _vel[i] = (_vel[i] + lin[i] * 0.00980665 * dt) * VEL_LEAK  # mg -> m/s^2
    _speed = math.sqrt(_vel[0] ** 2 + _vel[1] ** 2 + _vel[2] ** 2)
    if _speed > _speed_max:
        _speed_max = _speed
    return _speed


def build_meta(ts, w, h, frac, heartbeat, buffered):
    # "ip" is how ota_push.py finds the camera — keep it in every meta
    try:
        ip = network.WLAN(network.STA_IF).ifconfig()[0]
    except OSError:
        ip = ''
    return ('{"ts": %d, "w": %d, "h": %d, "diff": %.5f, '
            '"heartbeat": %s, "buffered": %s, "v": "%s", "ip": "%s"}'
            % (ts, w, h, frac,
               'true' if heartbeat else 'false',
               'true' if buffered else 'false', APP_VERSION, ip))


def post_jpeg(data, meta):
    """Minimal HTTP POST of a JPEG + X-Meta JSON. True on 2xx."""
    s = socket.socket()
    s.settimeout(10)
    try:
        s.connect(socket.getaddrinfo(secrets.SERVER_HOST, secrets.SERVER_PORT)[0][-1])
        head = ('POST /blob/%s/frame HTTP/1.1\r\n'
                'Host: %s\r\n'
                'Content-Type: image/jpeg\r\n'
                'X-Meta: %s\r\n'
                'Content-Length: %d\r\n'
                'Connection: close\r\n\r\n'
                % (secrets.DEVICE, secrets.SERVER_HOST, meta, len(data)))
        s.write(head.encode())
        mv = memoryview(data)
        for i in range(0, len(data), 4096):
            s.write(mv[i:i + 4096])
        resp = s.read(15)
        return b' 2' in resp[:13]
    except OSError:
        return False
    finally:
        s.close()


def telemetry(wlan, stats, t_boot, pending):
    vals = {
        'heap_free': gc.mem_free(),
        'uptime_s': time.ticks_diff(time.ticks_ms(), t_boot) // 1000,
        'frames_sent': stats[0],
        'frames_skipped': stats[1],
        'upload_failures': stats[2],
        'pending_files': pending,
    }
    for key, fn in (('batt_v', lambda: round(batt_volts(), 3)),
                    ('charging', lambda: 0 if _chg_pin.value() else 1),  # active-low
                    ('rssi', lambda: wlan.status('rssi'))):
        try:
            vals[key] = fn()
        except (OSError, ValueError):
            pass
    global _speed_max
    vals['speed_mps'] = round(_speed, 2)
    vals['speed_max_mps'] = round(_speed_max, 2)
    _speed_max = 0.0                     # max is per telemetry interval
    return vals


def mqtt_publish(vals):
    """Publish telemetry dict; persistent client, rebuilt on any failure."""
    global _mq
    try:
        if _mq is None:
            _mq = mqtt.MQTTClient('n6cam', secrets.SERVER_HOST, port=1883,
                                  user=secrets.MQTT_USER, password=secrets.MQTT_PASS)
            _mq.connect()
        for k, v in vals.items():
            _mq.publish(('%s/%s' % (secrets.MQTT_TOPIC, k)).encode(),
                        ('{"v": %s}' % v).encode())
        return True
    except Exception:
        try:
            _mq.disconnect()
        except Exception:
            pass
        _mq = None
        return False


# --- SD offline buffer ------------------------------------------------------

def sd_ready():
    try:
        os.stat('/sdcard')
        try:
            os.mkdir(PENDING_DIR)
        except OSError:
            pass
        os.stat(PENDING_DIR)
        return True
    except OSError:
        return False


def pending_count():
    try:
        return sum(1 for f in os.listdir(PENDING_DIR) if f.endswith('.jpg'))
    except OSError:
        return 0


def buffer_frame(data, meta):
    """Write meta sidecar first, then tmp-rename the jpg — a .jpg on disk
    always has its .json. Returns True on success."""
    base = '%s/%010d' % (PENDING_DIR, time.time())
    i = 0
    name = base
    while True:
        try:
            os.stat(name + '.jpg')
            i += 1
            name = '%s_%d' % (base, i)
        except OSError:
            break
    try:
        with open(name + '.json', 'w') as f:
            f.write(meta)
        with open(name + '.tmp', 'wb') as f:
            f.write(data)
        os.rename(name + '.tmp', name + '.jpg')
        return True
    except OSError:
        for ext in ('.json', '.tmp'):
            try:
                os.remove(name + ext)
            except OSError:
                pass
        return False


def drain_pending(poll):
    """Upload the whole backlog oldest-first. Recording is paused by the
    caller for the duration. Returns True when the dir is empty."""
    try:
        files = sorted(f for f in os.listdir(PENDING_DIR) if f.endswith('.jpg'))
    except OSError:
        return True
    for f in files:
        base = PENDING_DIR + '/' + f[:-4]
        try:
            with open(base + '.jpg', 'rb') as fp:
                data = fp.read()
            try:
                with open(base + '.json') as fp:
                    meta = fp.read()
            except OSError:
                meta = '{"buffered": true, "v": "%s"}' % APP_VERSION
        except OSError:
            continue
        if not post_jpeg(data, meta):
            return False                    # server gone again; resume offline
        for ext in ('.jpg', '.json'):
            try:
                os.remove(base + ext)
            except OSError:
                pass
        sensor.snapshot()                   # keep the live stream fresh
        poll()                              # keep OTA responsive mid-drain
    return True


# --- main loop --------------------------------------------------------------

def run(poll=lambda: None):
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)
    sensor.skip_frames(time=1500)
    w, h = sensor.width(), sensor.height()

    ref = image.Image(w, h, sensor.RGB565)   # last recorded frame
    work = image.Image(w, h, sensor.RGB565)  # scratch for the diff
    have_ref = False

    wlan = network.WLAN(network.STA_IF)
    sd_buf = sd_ready()
    pending = pending_count() if sd_buf else 0
    print('sd buffer:', 'ready, %d pending' % pending if sd_buf else 'no card')

    sent = failed = skipped = buffered = 0
    last_upload = last_telemetry = last_wifi_try = last_batt_watch = last_speed_watch = 0
    t_boot = time.ticks_ms()

    while True:
        t0 = time.ticks_ms()
        try:
            if time.time() - last_telemetry >= TELEMETRY_S:
                last_telemetry = time.time()
                vals = telemetry(wlan, (sent, skipped, failed), t_boot, pending)
                print('telemetry %s  batt=%sV rssi=%s pending=%d' %
                      ('OK' if mqtt_publish(vals) else 'FAIL',
                       vals.get('batt_v'), vals.get('rssi'), pending))

            img = sensor.snapshot()
            frac = 1.0
            if have_ref:
                work.replace(img)
                work.difference(ref)
                frac = sum(work.get_histogram().l_bins()[DIFF_L_THRESH:])
            heartbeat = time.time() - last_upload >= HEARTBEAT_S

            if frac >= DIFF_MIN_FRAC or heartbeat:
                ref.replace(img)
                have_ref = True
                ts = time.time() + EPOCH_OFFSET
                jpg = img.to_jpeg(quality=85)
                data = jpg.bytearray()

                if not wlan.isconnected() and time.time() - last_wifi_try >= WIFI_RETRY_S:
                    last_wifi_try = time.time()
                    try:
                        wlan.active(True)
                        wlan.connect(secrets.WIFI_SSID, secrets.WIFI_PASS)
                        end = time.time() + 8
                        while not wlan.isconnected() and time.time() < end:
                            time.sleep_ms(300)
                    except OSError:
                        pass

                delivered = False
                if wlan.isconnected():
                    if pending:
                        # backlog exists: queue this frame too, then drain
                        # everything in order (recording pauses during drain)
                        meta = build_meta(ts, w, h, frac, heartbeat, True)
                        if sd_buf and pending < MAX_PENDING and buffer_frame(data, meta):
                            pending += 1
                            buffered += 1
                        print('draining %d pending...' % pending)
                        if drain_pending(poll):
                            sent += pending
                            pending = 0
                            last_upload = time.time()
                            delivered = True
                            print('drain complete')
                            if time.localtime()[0] < 2021:
                                try:
                                    import ntptime
                                    ntptime.settime()
                                    print('ntp: late sync')
                                except Exception:
                                    pass
                        else:
                            pending = pending_count()
                            print('drain interrupted, %d left' % pending)
                    else:
                        meta = build_meta(ts, w, h, frac, heartbeat, False)
                        if post_jpeg(data, meta):
                            sent += 1
                            last_upload = time.time()
                            delivered = True

                if not delivered:
                    failed += 1
                    if sd_buf and pending < MAX_PENDING:
                        meta = build_meta(ts, w, h, frac, heartbeat, True)
                        if buffer_frame(data, meta):
                            pending += 1
                            buffered += 1
                            last_upload = time.time()   # buffered counts for heartbeat timer
                print('%s  diff=%.5f  sent=%d skipped=%d failed=%d buffered=%d pending=%d' %
                      ('upload OK' if delivered else 'offline->sd',
                       frac, sent, skipped, failed, buffered, pending))
            else:
                skipped += 1
                print('skip  diff=%.5f  sent=%d skipped=%d pending=%d' %
                      (frac, sent, skipped, pending))
        except Exception as e:
            failed += 1
            print('loop error:', repr(e))
        # keep live snapshots flowing (stream viewer stays fresh) and the OTA
        # listener responsive between checks
        while time.ticks_diff(time.ticks_ms(), t0) < secrets.PERIOD_S * 1000:
            sensor.snapshot()
            poll()
            if BATT_WATCH_S and time.time() - last_batt_watch >= BATT_WATCH_S:
                last_batt_watch = time.time()
                try:
                    mqtt_publish({'batt_v': round(batt_volts(), 3),
                                  'charging': 0 if _chg_pin.value() else 1})
                except Exception:
                    pass
            if SPEED_WATCH_S and time.time() - last_speed_watch >= SPEED_WATCH_S:
                last_speed_watch = time.time()
                try:
                    mqtt_publish({'speed_mps': round(_speed, 2)})
                except Exception:
                    pass
            # ~50 Hz IMU sampling between frame checks
            for _ in range(5):
                try:
                    imu_speed_sample()
                except (OSError, ValueError):
                    pass
                time.sleep_ms(20)
