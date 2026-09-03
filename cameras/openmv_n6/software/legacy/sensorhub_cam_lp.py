"""Low-power variant of the N6 sensorhub camera. Push with:
    ./ota_push.py sensorhub_cam_lp.py --wait
(--wait retries until the camera's next WiFi window; see below.)

Power strategy vs the always-on app (sensorhub_cam.py):
- Sensor runs at SENSOR_FPS (5) instead of the free-running ~117 fps ISP,
  and snapshots happen once per check instead of 10x/s. NO LIVE STREAM —
  fb_webui.py will show a stale frame in this mode.
- Motion check every CHECK_PERIOD_S (60 s). WiFi is OFF between deliveries:
  it connects only when there is something to send (motion, heartbeat,
  backlog), lingers WIFI_LINGER_S (15 s) for OTA pushes, then drops. The
  HEARTBEAT_S (300 s) heartbeat guarantees a window at least every 5 min —
  that is the OTA/debug access cadence.
- CPU clock lowered via cpufreq if supported (guarded; first boot of this
  app should be supervised near USB — see the ADCAll hard-hang lesson).
- Telemetry batches into each WiFi window instead of a fixed 60 s timer.
- SD offline buffering + oldest-first drain carried over unchanged.

Expected draw ~80-150 mA vs ~300 mA always-on (measure to confirm).
Everything else (raw-socket POST, heap buffers, secrets-not-OTA-managed)
follows the constraints documented in sensorhub_cam.py.
"""
import gc
import os
import socket
import time

import image
import machine
import mqtt
import network
import sensor

import ota
import secrets

APP_VERSION = 'lp-1.0'

EPOCH_OFFSET = 946684800 if time.time() < 1_000_000_000 else 0  # 2000 vs 1970 epoch

DIFF_MIN_FRAC = getattr(secrets, 'DIFF_MIN_FRAC', 0.005)
DIFF_L_THRESH = getattr(secrets, 'DIFF_L_THRESH', 8)
HEARTBEAT_S = getattr(secrets, 'HEARTBEAT_S', 300)
MAX_PENDING = getattr(secrets, 'MAX_PENDING', 2000)

CHECK_PERIOD_S = 60      # motion-check cadence (was PERIOD_S=10 always-on)
WIFI_LINGER_S = 15       # OTA window after each delivery
SENSOR_FPS = 5           # sensor/ISP rate (always-on mode free-runs ~117)
CPU_MHZ = 0              # underclock target; 0 = leave alone. Keep 0 until a
                         # SUPERVISED near-USB boot proves set_frequency safe
                         # alongside the CSI (see ADCAll hard-hang lesson).

PENDING_DIR = '/sdcard/pending'
BATT_DIVIDER = 1.5

_bat_adc = machine.ADC(machine.Pin.board.BAT_ADC)
_chg_pin = machine.Pin(machine.Pin.board.CHG, machine.Pin.IN, machine.Pin.PULL_UP)
_mq = None


def batt_volts():
    raw = sum(_bat_adc.read_u16() for _ in range(8)) // 8
    return raw / 65535 * 3.3 * BATT_DIVIDER


def build_meta(ts, w, h, frac, heartbeat, buffered):
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
                    ('charging', lambda: 0 if _chg_pin.value() else 1),
                    ('rssi', lambda: wlan.status('rssi'))):
        try:
            vals[key] = fn()
        except (OSError, ValueError):
            pass
    return vals


def mqtt_publish(vals):
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


# --- WiFi window ------------------------------------------------------------

def wifi_up(wlan, timeout_s=12):
    if wlan.isconnected():
        return True
    try:
        wlan.active(True)
        wlan.connect(secrets.WIFI_SSID, secrets.WIFI_PASS)
    except OSError:
        return False
    end = time.time() + timeout_s
    while not wlan.isconnected() and time.time() < end:
        time.sleep_ms(300)
    return wlan.isconnected()


def wifi_down(wlan):
    global _mq
    try:
        _mq and _mq.disconnect()
    except Exception:
        pass
    _mq = None
    try:
        wlan.disconnect()
        wlan.active(False)          # radio off between windows
    except OSError:
        pass


# --- SD offline buffer (same scheme as always-on app) -----------------------

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
            return False
        for ext in ('.jpg', '.json'):
            try:
                os.remove(base + ext)
            except OSError:
                pass
        poll()
    return True


# --- main loop --------------------------------------------------------------

def run(poll=lambda: None):
    try:
        import cpufreq
        if CPU_MHZ:
            cpufreq.set_frequency(CPU_MHZ)
        print('cpu clock:', cpufreq.get_frequency() if hasattr(cpufreq, 'get_frequency') else CPU_MHZ)
    except Exception as e:
        print('cpufreq unavailable:', repr(e))

    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)
    sensor.set_framerate(SENSOR_FPS)
    sensor.skip_frames(time=1500)
    w, h = sensor.width(), sensor.height()

    ref = image.Image(w, h, sensor.RGB565)
    work = image.Image(w, h, sensor.RGB565)
    have_ref = False

    wlan = network.WLAN(network.STA_IF)
    sd_buf = sd_ready()
    pending = pending_count() if sd_buf else 0
    print('lp mode: check every %ds, wifi windows only; sd buffer: %s' %
          (CHECK_PERIOD_S, '%d pending' % pending if sd_buf else 'no card'))

    sent = failed = skipped = 0
    last_upload = 0
    t_boot = time.ticks_ms()

    while True:
        t0 = time.ticks_ms()
        try:
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

                delivered = False
                if wifi_up(wlan):
                    if pending:
                        meta = build_meta(ts, w, h, frac, heartbeat, True)
                        if sd_buf and pending < MAX_PENDING and buffer_frame(data, meta):
                            pending += 1
                        if drain_pending(poll):
                            sent += pending
                            pending = 0
                            last_upload = time.time()
                            delivered = True
                            if time.localtime()[0] < 2021:
                                try:
                                    import ntptime
                                    ntptime.settime()
                                except Exception:
                                    pass
                        else:
                            pending = pending_count()
                    else:
                        meta = build_meta(ts, w, h, frac, heartbeat, False)
                        if post_jpeg(data, meta):
                            sent += 1
                            last_upload = time.time()
                            delivered = True
                    mqtt_publish(telemetry(wlan, (sent, skipped, failed), t_boot, pending))
                    # linger with WiFi up so ota_push.py --wait can reach us
                    end = time.ticks_ms()
                    while time.ticks_diff(time.ticks_ms(), end) < WIFI_LINGER_S * 1000:
                        poll()
                        time.sleep_ms(100)
                    wifi_down(wlan)

                if not delivered:
                    failed += 1
                    if sd_buf and pending < MAX_PENDING:
                        meta = build_meta(ts, w, h, frac, heartbeat, True)
                        if buffer_frame(data, meta):
                            pending += 1
                            last_upload = time.time()   # buffered counts for heartbeat
                    wifi_down(wlan)
                print('%s  diff=%.5f  sent=%d skipped=%d failed=%d pending=%d' %
                      ('delivered' if delivered else 'offline->sd',
                       frac, sent, skipped, failed, pending))
            else:
                skipped += 1
                print('skip  diff=%.5f  sent=%d skipped=%d pending=%d' %
                      (frac, sent, skipped, pending))
        except Exception as e:
            failed += 1
            print('loop error:', repr(e))
        # low-power wait: no continuous snapshots, no radio; poll() still runs
        # so a USB-attached OTA push during a linger window stays possible
        while time.ticks_diff(time.ticks_ms(), t0) < CHECK_PERIOD_S * 1000:
            poll()
            time.sleep_ms(250)
