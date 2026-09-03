"""OpenMV RT1062 (R6) -> sensorhub uploader with motion gating, MQTT
telemetry, and SD offline buffering. Deployed as /flash/app.py, run by the
OTA bootstrap (ota_main.py as /flash/main.py) — update over WiFi with
./ota_push.py.

Same behaviour as cameras/openmv_n6/software/sensorhub_cam.py minus the
N6-only parts (IMU speed estimate, BAT_ADC divider, cpufreq). Kept as its
own file because the two boards run different OpenMV firmware generations
(RT1062: 4.8.1 / MicroPython 1.26; N6: 5.0 / 1.28).

Normal operation: every PERIOD_S seconds, diff a VGA RGB565 frame against
the last *recorded* frame; when enough pixels changed or HEARTBEAT_S passed,
switch the sensor to CAPTURE_FRAMESIZE (default WQXGA2 = 2592x1944, the
OV5640's native size) in JPEG mode, take one frame straight from the
sensor's hardware encoder, POST it to the sensorhub ingest, and drop back
to VGA. The full-res JPEG stays in the frame buffer (no 10 MB RGB565 copy,
no heap copy) until it has been uploaded or written to SD. If the mode
switch or snapshot fails, the frame is a software-encoded VGA JPEG as in
1.4-rt. capture_framesize = "" in config.toml disables the switch.

Focus mode (1.6-rt): for adjusting the lens in place. A second HTTP
listener on FOCUS_PORT (default 8267, no token unless secrets.FOCUS_TOKEN) serves a phone
page whose <img> is a VGA grayscale MJPEG stream with a live sharpness
score (stdev of a x4 Laplacian over the centre ROI) and the session's best
score drawn on every frame. Recording pauses for the session; it ends on
timeout (`secs`, default focus_secs), a GET /stop, the viewer staying
disconnected for 10 s, or a press of the user button — which also *starts*
a session (LED on, waiting for a viewer). Telemetry only goes out while
no viewer is attached (a TLS publish blocks the stream for seconds);
/status carries per-stage worst-case timings for diagnosing stalls. Telemetry (charging, RSSI, counters,
pending backlog) publishes over MQTT every TELEMETRY_S as
<DEVICE>/<variable>.

Offline: when WiFi or the server is unreachable, motion/heartbeat frames
are written to /sdcard/pending/ (JPEG + .json meta sidecar with the real
capture ts, tagged "buffered": true). WiFi rejoin is attempted every 30 s.
When an upload succeeds again, recording pauses and the entire backlog
drains oldest-first, then live operation resumes. No SD card -> buffering
quietly disabled.

Constraints carried over from the N6:
- Heap image.Image() buffers, not alloc_extra_fb.
- Raw-socket POST: frozen `requests` chokes on the ingest's HTTP/1.0 reply.
- secrets.py is NOT OTA-managed; new config knobs use getattr defaults.
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

import secrets

APP_VERSION = '1.6-rt'

EPOCH_OFFSET = 946684800 if time.time() < 1_000_000_000 else 0  # 2000 vs 1970 epoch

# --- tuning: generated from ~/.dusty/config.toml [camera.openmv_rt1062] by
# gen_secrets.py — lives in the app (OTA-updatable), NOT in secrets.py, so a
# threshold change is `./gen_secrets.py && ./ota_push.py sensorhub_cam.py`.
TUNING = {'period_s': 10, 'diff_min_frac': 0.04, 'diff_l_thresh': 20, 'heartbeat_s': 300, 'telemetry_s': 60, 'capture_framesize': 'WQXGA2', 'capture_settle_ms': 400, 'focus_secs': 300}
# --- end tuning
PERIOD_S = TUNING['period_s']
DIFF_MIN_FRAC = TUNING['diff_min_frac']
DIFF_L_THRESH = TUNING['diff_l_thresh']
HEARTBEAT_S = TUNING['heartbeat_s']
TELEMETRY_S = TUNING['telemetry_s']
CAPTURE_FRAMESIZE = TUNING.get('capture_framesize', 'WQXGA2')  # sensor.<NAME>; '' = VGA only
CAPTURE_SETTLE_MS = TUNING.get('capture_settle_ms', 400)       # AE/AWB settle after the switch
FOCUS_SECS = TUNING.get('focus_secs', 300)                     # default focus session length
FOCUS_PORT = getattr(secrets, 'FOCUS_PORT', 8267)
FOCUS_TOKEN = getattr(secrets, 'FOCUS_TOKEN', '')   # unset = open on the LAN (it is a viewfinder)
FOCUS_ROI = (200, 120, 240, 240)   # centre of VGA, where the score is measured
FOCUS_JPEG_Q = 60
MAX_PENDING = getattr(secrets, 'MAX_PENDING', 2000)
WIFI_RETRY_S = getattr(secrets, 'WIFI_RETRY_S', 30)
JPEG_QUALITY = getattr(secrets, 'JPEG_QUALITY', 85)
SERVER_TLS = getattr(secrets, 'SERVER_TLS', False)      # HTTPS (Tailscale Funnel path)
BLOB_TOKEN = getattr(secrets, 'BLOB_TOKEN', '')         # X-Token for the public gate

PENDING_DIR = '/sdcard/pending'

# RT1062 R6 exposes CHG (active-low charge indicator) but no BAT_ADC pin
try:
    _chg_pin = machine.Pin(machine.Pin.board.CHG, machine.Pin.IN, machine.Pin.PULL_UP)
except (AttributeError, ValueError, OSError):
    _chg_pin = None
_mq = None


def build_meta(ts, w, h, frac, heartbeat, buffered):
    # "ip" is how ota_push.py finds the camera — keep it in every meta
    try:
        ip = network.WLAN(network.STA_IF).ifconfig()[0]
    except OSError:
        ip = ''
    return ('{"ts": %d, "w": %d, "h": %d, "diff": %.5f, "gate": %.4f, '
            '"heartbeat": %s, "buffered": %s, "v": "%s", "ip": "%s"}'
            % (ts, w, h, frac, DIFF_MIN_FRAC,
               'true' if heartbeat else 'false',
               'true' if buffered else 'false', APP_VERSION, ip))


def post_jpeg(data, meta, length=None):
    """Minimal HTTP(S) POST of a JPEG + X-Meta JSON. True on 2xx.

    `data` is bytes-like, or an open file (pass `length`) which is streamed
    in 4 kB chunks so a full-res frame from SD never lands in the heap.

    TLS: no CA bundle on the board, so the server cert is not verified —
    the shared BLOB_TOKEN is what authenticates, the tunnel only hides it."""
    s = socket.socket()
    s.settimeout(15)
    try:
        s.connect(socket.getaddrinfo(secrets.SERVER_HOST, secrets.SERVER_PORT)[0][-1])
        if SERVER_TLS:
            import ssl
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.verify_mode = ssl.CERT_NONE
            s = ctx.wrap_socket(s, server_hostname=secrets.SERVER_HOST)
        head = ('POST /blob/%s/frame HTTP/1.1\r\n'
                'Host: %s\r\n'
                'Content-Type: image/jpeg\r\n'
                'X-Meta: %s\r\n'
                '%s'
                'Content-Length: %d\r\n'
                'Connection: close\r\n\r\n'
                % (secrets.DEVICE, secrets.SERVER_HOST, meta,
                   ('X-Token: %s\r\n' % BLOB_TOKEN) if BLOB_TOKEN else '',
                   len(data) if length is None else length))
        s.write(head.encode())
        if hasattr(data, 'read'):
            while True:
                chunk = data.read(4096)
                if not chunk:
                    break
                s.write(chunk)
        else:
            mv = memoryview(data)
            for i in range(0, len(data), 4096):
                s.write(mv[i:i + 4096])
        resp = s.read(15)
        return b' 2' in resp[:13]
    except OSError as e:
        print('post error:', repr(e))
        return False
    finally:
        s.close()


def http_publish(vals):
    """Telemetry fallback for the public path: POST JSON to /telemetry/<device>
    on the blob gate, which republishes each key as <device>/<key> on MQTT."""
    body = ('{%s}' % ', '.join('"%s": %s' % kv for kv in vals.items())).encode()
    s = socket.socket()
    s.settimeout(15)
    try:
        s.connect(socket.getaddrinfo(secrets.SERVER_HOST, secrets.SERVER_PORT)[0][-1])
        if SERVER_TLS:
            import ssl
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.verify_mode = ssl.CERT_NONE
            s = ctx.wrap_socket(s, server_hostname=secrets.SERVER_HOST)
        s.write(('POST /telemetry/%s HTTP/1.1\r\nHost: %s\r\n'
                 'Content-Type: application/json\r\n%sContent-Length: %d\r\n'
                 'Connection: close\r\n\r\n'
                 % (secrets.DEVICE, secrets.SERVER_HOST,
                    ('X-Token: %s\r\n' % BLOB_TOKEN) if BLOB_TOKEN else '', len(body))).encode())
        s.write(body)
        resp = s.read(15)
        return b' 2' in resp[:13]
    except OSError as e:
        print('telemetry post error:', repr(e))
        return False
    finally:
        s.close()


def publish(vals):
    """Telemetry goes through the blob gate over HTTP(S) by default — one
    code path at home and remote. Direct MQTT (TELEMETRY_MQTT = True in
    secrets.py) needs a broker user allowed to write <DEVICE>/#; a QoS-0
    publish the ACL rejects is dropped silently and would still look OK."""
    if getattr(secrets, 'TELEMETRY_MQTT', False):
        return mqtt_publish(vals)
    return http_publish(vals)


_diff_stats = [0.0, 0.0, 0]   # max diff, sum, count over the telemetry window


def note_diff(frac):
    _diff_stats[0] = max(_diff_stats[0], frac)
    _diff_stats[1] += frac
    _diff_stats[2] += 1


def telemetry(wlan, stats, t_boot, pending):
    vals = {
        'heap_free': gc.mem_free(),
        'uptime_s': time.ticks_diff(time.ticks_ms(), t_boot) // 1000,
        'frames_sent': stats[0],
        'frames_skipped': stats[1],
        'upload_failures': stats[2],
        'pending_files': pending,
        # motion-gate tuning aids: what the diff looked like this window
        'diff_max': round(_diff_stats[0], 4),
        'diff_mean': round(_diff_stats[1] / _diff_stats[2], 4) if _diff_stats[2] else 0,
        'diff_gate': DIFF_MIN_FRAC,
    }
    _diff_stats[0] = _diff_stats[1] = 0.0
    _diff_stats[2] = 0
    vals['focus'] = 1 if _focus['active'] else 0
    if _focus['active']:
        vals['focus_score'] = round(_focus['score'], 1)
        vals['focus_best'] = round(_focus['best'], 1)
    if _chg_pin is not None:
        try:
            vals['charging'] = 0 if _chg_pin.value() else 1   # active-low
        except (OSError, ValueError):
            pass
    try:
        vals['rssi'] = wlan.status('rssi')
    except (OSError, ValueError):
        pass
    return vals


def mqtt_publish(vals):
    """Publish telemetry dict; persistent client, rebuilt on any failure."""
    global _mq
    try:
        if _mq is None:
            _mq = mqtt.MQTTClient(secrets.DEVICE, secrets.SERVER_HOST, port=1883,
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
            try:
                with open(base + '.json') as fp:
                    meta = fp.read()
            except OSError:
                meta = '{"buffered": true, "v": "%s"}' % APP_VERSION
            with open(base + '.jpg', 'rb') as fp:
                ok = post_jpeg(fp, meta, os.stat(base + '.jpg')[6])
        except OSError:
            continue
        if not ok:
            return False                    # server gone again; resume offline
        for ext in ('.jpg', '.json'):
            try:
                os.remove(base + ext)
            except OSError:
                pass
        sensor.snapshot()                   # keep the live stream fresh
        poll()                              # keep OTA responsive mid-drain
    return True


# --- focus mode ---------------------------------------------------------------

_focus = {'active': False, 'score': 0.0, 'best': 0.0, 'until': 0,
          'stop': False, 'sessions': 0, 'frames': 0,
          # diagnostics: request log + worst-case ms per stage of the stream loop
          'reqs': 0, 'req_timeouts': 0, 'last_req': '', 'last_end': '', 'viewer': False,
          'ms_poll': 0, 'ms_tick': 0, 'ms_accept': 0, 'ms_frame': 0, 'ms_write': 0}
_focus_srv = None
_btn = None
_btn_name = ''
_led = None


def _init_focus():
    """Listener + button + LED. All optional: a failure just logs."""
    global _focus_srv, _btn, _btn_name, _led
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', FOCUS_PORT))
        s.listen(1)
        s.setblocking(False)
        _focus_srv = s
        print('focus: listening on :%d' % FOCUS_PORT)
    except OSError as e:
        print('focus: listener failed:', repr(e))
    # user button: pin name differs per firmware; probe, then report via /status
    for name in ('USR_BTN', 'BTN', 'USR', 'BOOT', 'SW', 'SW1', 'BUTTON', 'USER_BTN'):
        try:
            _btn = machine.Pin(getattr(machine.Pin.board, name), machine.Pin.IN, machine.Pin.PULL_UP)
            _btn_name = name
            print('focus: button on', name)
            break
        except (AttributeError, ValueError, OSError):
            pass
    if _btn is None:
        print('focus: no user button found; URL only')
    try:
        _led = machine.LED('LED_BLUE')
    except Exception:
        try:
            import pyb
            _led = pyb.LED(3)
        except Exception:
            _led = None


def _led_set(on):
    if _led is None:
        return
    try:
        _led.on() if on else _led.off()
    except Exception:
        pass


_btn_was = False


def _btn_edge():
    """True once per press: active-low and held for 300 ms (glitch-proof)."""
    global _btn_was
    if _btn is None:
        return False
    down = _btn.value() == 0
    if down and not _btn_was:
        for _ in range(6):
            time.sleep_ms(50)
            if _btn.value() != 0:
                down = False
                break
    edge = down and not _btn_was
    _btn_was = down
    return edge


def _board_pins():
    names = []
    try:
        for n in dir(machine.Pin.board):
            u = n.upper()
            if 'BTN' in u or 'SW' in u or 'USR' in u or 'BOOT' in u or 'LED' in u or 'BUT' in u:
                names.append(n)
    except Exception:
        pass
    return names


def _http_reply(conn, code, ctype, body):
    if isinstance(body, str):
        body = body.encode()
    conn.write(('HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n'
                'Cache-Control: no-cache\r\nConnection: close\r\n\r\n'
                % (code, ctype, len(body))).encode())
    conn.write(body)


def _focus_accept():
    """Non-blocking accept + request-line parse -> (conn, path, query) or None."""
    if _focus_srv is None:
        return None
    try:
        conn, _ = _focus_srv.accept()
    except OSError:
        return None
    _focus['reqs'] += 1
    try:
        conn.setblocking(True)
        conn.settimeout(1)                  # a silent connection must not stall the stream
        line = conn.readline()
        while True:                         # drain headers
            h = conn.readline()
            if not h or h == b'\r\n':
                break
        parts = line.split()
        target = parts[1].decode() if len(parts) >= 2 else '/'
        path, _, qs = target.partition('?')
        query = {}
        for kv in qs.split('&'):
            k, _, v = kv.partition('=')
            if k:
                query[k] = v
        _focus['last_req'] = target
        return conn, path, query
    except Exception as e:
        _focus['req_timeouts'] += 1
        print('focus: bad request', repr(e))
        conn.close()
        return None


def _focus_status():
    rem = max(0, _focus['until'] - time.time()) if _focus['active'] else 0
    return ('{"focus": %s, "score": %.1f, "best": %.1f, "remaining_s": %d, '
            '"sessions": %d, "frames": %d, "button": "%s", "pins": %s, "v": "%s", '
            '"reqs": %d, "req_timeouts": %d, "last_req": "%s", "last_end": "%s", "viewer": %s, '
            '"max_ms": {"poll": %d, "tick": %d, "accept": %d, "frame": %d, "write": %d}}'
            % ('true' if _focus['active'] else 'false', _focus['score'], _focus['best'],
               rem, _focus['sessions'], _focus['frames'], _btn_name,
               '[%s]' % ', '.join('"%s"' % n for n in _board_pins()), APP_VERSION,
               _focus['reqs'], _focus['req_timeouts'], _focus['last_req'],
               _focus['last_end'], 'true' if _focus['viewer'] else 'false',
               _focus['ms_poll'], _focus['ms_tick'], _focus['ms_accept'],
               _focus['ms_frame'], _focus['ms_write']))


def _focus_page(query):
    tok = query.get('token', '')
    q = ('token=%s&' % tok) if tok else ''
    secs = query.get('secs', str(FOCUS_SECS))
    return ('<!doctype html><title>%s focus</title>'
            '<meta name=viewport content="width=device-width,initial-scale=1">'
            '<body style="margin:0;background:#000;color:#eee;font:16px sans-serif;text-align:center">'
            '<img id=s src="/stream?%ssecs=%s" style="width:100%%;display:block">'
            '<p id=m>turn the lens until the score peaks</p>'
            '<p><a style="color:#8cf" href="/?%ssecs=%s">restart</a> &middot; '
            '<a style="color:#f88" href="/stop%s">stop</a></p>'
            # watchdog: if the camera stops counting frames, or the session ended,
            # re-open the stream (phones drop long multipart connections)
            '<script>var Q="%s",S="%s",last=-1,same=0;'
            'function re(){document.getElementById("s").src="/stream?"+Q+"secs="+S+"&t="+Date.now();same=0;}'
            'setInterval(function(){fetch("/status").then(function(r){return r.json()}).then(function(d){'
            'var m=document.getElementById("m");'
            'if(!d.focus){m.textContent="session ended ("+d.last_end+") - reconnecting";re();return;}'
            'same=(d.frames===last)?same+1:0;last=d.frames;'
            'm.textContent="score "+d.score+"  best "+d.best+"  "+d.remaining_s+"s left  "+(d.viewer?"":"(no viewer) ");'
            'if(same>=2){m.textContent+="- stream stalled, reconnecting";re();}'
            '}).catch(function(){})},2000);</script>'
            % (secrets.DEVICE, q, secs, q, secs, ('?' + q.rstrip('&')) if q else '', q, secs))


def _authed(query):
    return not FOCUS_TOKEN or query.get('token', '') == FOCUS_TOKEN


def _focus_dispatch(req, poll, tick, in_session):
    """Handle one request. Outside a session may run one (returns True).
    Inside a session returns a new viewer conn for /stream, else None."""
    conn, path, query = req
    try:
        if path == '/status':
            _http_reply(conn, '200 OK', 'application/json', _focus_status())
        elif not _authed(query):
            _http_reply(conn, '403 Forbidden', 'application/json', '{"error": "bad token"}')
        elif path in ('/', '/focus'):
            _http_reply(conn, '200 OK', 'text/html', _focus_page(query))
        elif path == '/stop':
            _focus['stop'] = True
            _http_reply(conn, '200 OK', 'text/html',
                        '<!doctype html><body style="background:#000;color:#eee;font:20px sans-serif;'
                        'text-align:center"><p>focus mode %s; recording resumes.</p>'
                        % ('stopping' if in_session else 'was not active'))
        elif path == '/stream':
            try:
                secs = int(query.get('secs', FOCUS_SECS))
            except ValueError:
                secs = FOCUS_SECS
            conn.write(b'HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n'
                       b'Cache-Control: no-cache\r\nConnection: close\r\n\r\n')
            if in_session:
                _focus['until'] = max(_focus['until'], time.time() + secs)
                return conn                  # caller takes over the socket
            focus_session(conn, secs, poll, tick, 'url')
            return True
        else:
            _http_reply(conn, '404 Not Found', 'application/json', '{"error": "unknown"}')
    except OSError as e:
        print('focus: reply error', repr(e))
    conn.close()
    return None


def _focus_frame(remaining):
    """Grab a grayscale VGA frame, score the centre ROI, annotate -> jpeg bytes."""
    img = sensor.snapshot()
    crop = img.copy(roi=FOCUS_ROI)
    crop.laplacian(1, mul=4.0)          # x4 so the score is not stuck on integers
    st = crop.get_statistics()
    score = st.l_stdev() if hasattr(st, 'l_stdev') else st.stdev()
    _focus['score'] = score
    if score > _focus['best']:
        _focus['best'] = score
    img.draw_rectangle(FOCUS_ROI, color=255, thickness=2)
    txt = 'focus %5.1f   best %5.1f   %3ds' % (score, _focus['best'], remaining)
    img.draw_string(10, 10, txt, color=0, scale=4, mono_space=False)
    img.draw_string(8, 8, txt, color=255, scale=4, mono_space=False)
    bar = int(min(score, 100) / 100 * (img.width() - 16))
    img.draw_rectangle(8, img.height() - 24, bar, 16, color=255, fill=True)
    return img.to_jpeg(quality=FOCUS_JPEG_Q).bytearray()


def focus_session(conn, secs, poll, tick, trigger):
    """Run focus mode until timeout/stop/button/viewer gone. Blocks the
    recording loop; OTA poll and telemetry keep running via poll/tick."""
    global _fullres
    _focus['active'] = True
    _focus['stop'] = False
    _focus['best'] = 0.0
    _focus['score'] = 0.0
    _focus['until'] = time.time() + secs
    _focus['sessions'] += 1
    viewer_lost_at = None if conn else time.time()
    frames = 0
    for k in ('ms_poll', 'ms_tick', 'ms_accept', 'ms_frame', 'ms_write'):
        _focus[k] = 0

    def timed(key, fn, *a):
        t = time.ticks_ms()
        r = fn(*a)
        d = time.ticks_diff(time.ticks_ms(), t)
        if d > _focus[key]:
            _focus[key] = d
        return r
    print('focus: session start (%s, %ds)' % (trigger, secs))
    _led_set(True)
    try:
        restore_preview()
        sensor.set_pixformat(sensor.GRAYSCALE)
        sensor.set_framesize(sensor.VGA)
        sensor.skip_frames(time=300)
        while True:
            now = time.time()
            if _focus['stop']:
                _focus['last_end'] = 'stop'
                break
            if now >= _focus['until']:
                _focus['last_end'] = 'timeout'
                break
            if _btn_edge():
                _focus['last_end'] = 'button'
                print('focus: button -> stop')
                break
            # viewer that started by URL and went away: end after a grace period
            if conn is None and trigger == 'url' and viewer_lost_at and now - viewer_lost_at > 30:
                _focus['last_end'] = 'viewer gone'
                print('focus: viewer gone')
                break
            _focus['viewer'] = conn is not None
            timed('ms_poll', poll)
            if conn is None:
                timed('ms_tick', tick)      # telemetry (TLS, seconds) only while nobody is watching
            req = timed('ms_accept', _focus_accept)
            if req:
                new = _focus_dispatch(req, poll, tick, True)
                if new is not None and new is not True:
                    if conn:
                        conn.close()
                    conn = new
                    viewer_lost_at = None
            if conn is None:
                time.sleep_ms(50)
                continue
            try:
                data = timed('ms_frame', _focus_frame, int(_focus['until'] - now))
                t = time.ticks_ms()
                conn.write(('--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(data)).encode())
                mv = memoryview(data)
                for i in range(0, len(data), 4096):
                    conn.write(mv[i:i + 4096])
                conn.write(b'\r\n')
                d = time.ticks_diff(time.ticks_ms(), t)
                if d > _focus['ms_write']:
                    _focus['ms_write'] = d
                frames += 1
                _focus['frames'] += 1
            except OSError as e:
                print('focus: viewer dropped', repr(e))
                conn.close()
                conn = None
                viewer_lost_at = time.time()
    except Exception as e:
        _focus['last_end'] = 'error ' + repr(e)
        print('focus: session error', repr(e))
    finally:
        if conn:
            try:
                conn.close()
            except OSError:
                pass
        _focus['active'] = False
        _focus['viewer'] = False
        _led_set(False)
        try:
            sensor.set_pixformat(sensor.RGB565)
            sensor.set_framesize(sensor.VGA)
            sensor.skip_frames(time=CAPTURE_SETTLE_MS)
        except Exception as e:
            print('focus: restore failed', repr(e))
        gc.collect()
    print('focus: session end, %d frames, best %.1f' % (frames, _focus['best']))


def focus_poll(poll, tick):
    """Call from the idle loop. Returns True if a session ran (the caller
    resets its motion reference and holds the heartbeat timer)."""
    if _btn_edge():
        print('focus: button -> start')
        focus_session(None, FOCUS_SECS, poll, tick, 'button')
        return True
    req = _focus_accept()
    if req:
        return _focus_dispatch(req, poll, tick, False) is True
    return False


# --- capture ------------------------------------------------------------------

_fullres = False    # sensor currently in CAPTURE_FRAMESIZE/JPEG mode


def restore_preview():
    """Back to the VGA RGB565 stream the motion diff runs on."""
    global _fullres
    if not _fullres:
        return
    _fullres = False
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)
    sensor.skip_frames(time=CAPTURE_SETTLE_MS)


def capture():
    """One recorded frame -> (jpeg bytes-like, w, h).

    Full-res path: switch to CAPTURE_FRAMESIZE in JPEG mode so the OV5640
    compresses on-chip and the frame buffer only holds the JPEG (~1-2 MB at
    2592x1944) instead of a 10 MB RGB565 frame. The returned buffer aliases
    the frame buffer: it is valid until the next sensor.snapshot() /
    restore_preview(), so upload or write it to SD before either.
    Fallback: VGA snapshot, software JPEG (the 1.4-rt behaviour)."""
    global _fullres
    fs = getattr(sensor, CAPTURE_FRAMESIZE, None) if CAPTURE_FRAMESIZE else None
    if fs is not None:
        try:
            gc.collect()
            sensor.set_pixformat(sensor.JPEG)
            sensor.set_framesize(fs)
            _fullres = True
            if hasattr(sensor, 'set_quality'):
                sensor.set_quality(JPEG_QUALITY)
            sensor.skip_frames(time=CAPTURE_SETTLE_MS)
            img = sensor.snapshot()
            data = img.bytearray()
            w, h = img.width(), img.height()
            if len(data) > 4096 and w * h > 640 * 480:
                return data, w, h
            print('full-res capture rejected: %dx%d %d bytes' % (w, h, len(data)))
        except Exception as e:
            print('full-res capture failed:', repr(e))
        restore_preview()
    img = sensor.snapshot()
    jpg = img.to_jpeg(quality=JPEG_QUALITY)
    return jpg.bytearray(), jpg.width(), jpg.height()


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
    print('tuning:', TUNING)

    sent = failed = skipped = buffered = 0
    last_upload = last_telemetry = last_wifi_try = 0
    t_boot = time.ticks_ms()
    _init_focus()

    def tick():
        nonlocal last_telemetry
        if time.time() - last_telemetry >= TELEMETRY_S:
            last_telemetry = time.time()
            vals = telemetry(wlan, (sent, skipped, failed), t_boot, pending)
            print('telemetry %s  rssi=%s pending=%d' %
                  ('OK' if publish(vals) else 'FAIL',
                   vals.get('rssi'), pending))

    while True:
        t0 = time.ticks_ms()
        try:
            tick()
            img = sensor.snapshot()
            frac = 1.0
            if have_ref:
                work.replace(img)
                work.difference(ref)
                frac = sum(work.get_histogram().l_bins()[DIFF_L_THRESH:])
                note_diff(frac)
            heartbeat = time.time() - last_upload >= HEARTBEAT_S

            if frac >= DIFF_MIN_FRAC or heartbeat:
                ref.replace(img)
                have_ref = True
                ts = time.time() + EPOCH_OFFSET
                # `data` aliases the frame buffer while _fullres: no snapshot
                # until it is posted or on SD (drain restores preview first)
                data, cw, ch = capture()
                nbytes = len(data)

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
                        meta = build_meta(ts, cw, ch, frac, heartbeat, True)
                        if sd_buf and pending < MAX_PENDING and buffer_frame(data, meta):
                            pending += 1
                            buffered += 1
                        restore_preview()
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
                        meta = build_meta(ts, cw, ch, frac, heartbeat, False)
                        if post_jpeg(data, meta):
                            sent += 1
                            last_upload = time.time()
                            delivered = True

                if not delivered:
                    failed += 1
                    if sd_buf and pending < MAX_PENDING:
                        meta = build_meta(ts, cw, ch, frac, heartbeat, True)
                        if buffer_frame(data, meta):
                            pending += 1
                            buffered += 1
                            last_upload = time.time()   # buffered counts for heartbeat timer
                restore_preview()
                print('%s  %dx%d %dkB  diff=%.5f  sent=%d skipped=%d failed=%d buffered=%d pending=%d' %
                      ('upload OK' if delivered else 'offline->sd', cw, ch, nbytes // 1024,
                       frac, sent, skipped, failed, buffered, pending))
            else:
                skipped += 1
                print('skip  diff=%.5f  sent=%d skipped=%d pending=%d' %
                      (frac, sent, skipped, pending))
        except Exception as e:
            failed += 1
            print('loop error:', repr(e))
            try:
                restore_preview()
            except Exception as e2:
                print('restore_preview failed:', repr(e2))
        # keep live snapshots flowing and the OTA/focus listeners responsive
        while time.ticks_diff(time.ticks_ms(), t0) < PERIOD_S * 1000:
            sensor.snapshot()
            poll()
            if focus_poll(poll, tick):
                have_ref = False                # scene/exposure changed: no false motion
                last_upload = time.time()       # and no catch-up heartbeat
                t0 = time.ticks_ms()
            time.sleep_ms(100)
