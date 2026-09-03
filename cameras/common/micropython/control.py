"""control: the board's control plane on ONE port (secrets.OTA_PORT, 8266).
Standard §3 (modes) and §4 ("Control plane on the board").

Open on the LAN:  GET /status  GET /setup (page)  GET /stream (MJPEG)
                  GET /live (end setup)  POST /shoot  POST /refresh
Token (x-token = secrets.OTA_TOKEN):  POST /update (push OTA, dev fast path)

Takes the port over from the loader's ota.py listener at start, so the
loader (never updated OTA) needs no change; ota_push.py keeps working.

The app wires HOOKS before control_init():
  'tick'    fn()          publish telemetry if due (only called with no viewer)
  'shoot'   fn() -> bool  capture + deliver one frame now (why=manual)
  'refresh' fn() -> list  pull config (+ firmware check); changed keys
  'status'  fn() -> dict  extra status fields (version, cfg, counters)
  'sensors' fn() -> dict  live readings for the page
"""
import gc
import socket
import time

import machine
import secrets

from config import *
from camera import *
from focus import *
from otapull import *

CONTROL_PORT = getattr(secrets, 'OTA_PORT', 8266)
OTA_TOKEN = getattr(secrets, 'OTA_TOKEN', '')
MODE_NUM = {'live': 0, 'setup': 1, 'recovery': 2}

HOOKS = {}
STATE = {'mode': 'live', 'score': 0.0, 'best': 0.0, 'until': 0, 'stop': False,
         'sessions': 0, 'frames': 0, 'viewer': False,
         'reqs': 0, 'req_timeouts': 0, 'last_req': '', 'last_end': '',
         'ms_poll': 0, 'ms_tick': 0, 'ms_accept': 0, 'ms_frame': 0, 'ms_write': 0}
_srv = None
_btn = None
_btn_name = ''
_btn_was = False
_led = None
_t0 = time.ticks_ms()


def control_init(button_names=(), led_name='LED_BLUE'):
    global _srv, _btn, _btn_name, _led
    try:                                    # take the port over from the loader
        import ota
        if getattr(ota, '_srv', None):
            ota._srv.close()
            ota._srv = None
    except Exception:
        pass
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', CONTROL_PORT))
        s.listen(1)
        s.setblocking(False)
        _srv = s
        print('control: listening on :%d' % CONTROL_PORT)
    except OSError as e:
        print('control: listener failed:', repr(e))
    for name in button_names:
        try:
            _btn = machine.Pin(getattr(machine.Pin.board, name), machine.Pin.IN, machine.Pin.PULL_UP)
            _btn_name = name
            break
        except (AttributeError, ValueError, OSError):
            pass
    print('control: button', _btn_name or 'none')
    try:
        _led = machine.LED(led_name)
    except Exception:
        _led = None


def _led_set(on):
    if _led is None:
        return
    try:
        _led.on() if on else _led.off()
    except Exception:
        pass


def _btn_edge():
    """True once per press: active-low, held 300 ms."""
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


def _reply(conn, code, ctype, body):
    if isinstance(body, str):
        body = body.encode()
    conn.write(('HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n'
                'Cache-Control: no-cache\r\nConnection: close\r\n\r\n'
                % (code, ctype, len(body))).encode())
    conn.write(body)


def _accept():
    """Non-blocking accept + request parse -> (conn, method, path, query,
    headers) or None. A silent connection is dropped after 1 s."""
    if _srv is None:
        return None
    try:
        conn, _ = _srv.accept()
    except OSError:
        return None
    STATE['reqs'] += 1
    try:
        conn.setblocking(True)
        conn.settimeout(1)
        line = conn.readline()
        headers = {}
        while True:
            h = conn.readline()
            if not h or h == b'\r\n':
                break
            k, _, v = h.decode().partition(':')
            headers[k.strip().lower()] = v.strip()
        parts = line.split()
        method = parts[0].decode() if parts else 'GET'
        target = parts[1].decode() if len(parts) >= 2 else '/'
        path, _, qs = target.partition('?')
        query = {}
        for kv in qs.split('&'):
            k, _, v = kv.partition('=')
            if k:
                query[k] = v
        STATE['last_req'] = target
        return conn, method, path, query, headers
    except Exception as e:
        STATE['req_timeouts'] += 1
        print('control: bad request', repr(e))
        conn.close()
        return None


def _j(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return '%s' % v
    if isinstance(v, dict):
        return '{%s}' % ', '.join('"%s": %s' % (k, _j(x)) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return '[%s]' % ', '.join(_j(x) for x in v)
    return '"%s"' % v


def status_dict():
    d = {'mode': STATE['mode'], 'uptime_s': time.ticks_diff(time.ticks_ms(), _t0) // 1000,
         'cfg': CFG.get('cfg', 0), 'mem_free': gc.mem_free(), 'button': _btn_name,
         'setup': {'score': STATE['score'], 'best': STATE['best'],
                   'remaining_s': max(0, STATE['until'] - time.time()) if STATE['mode'] == 'setup' else 0,
                   'sessions': STATE['sessions'], 'frames': STATE['frames'],
                   'viewer': STATE['viewer'], 'last_end': STATE['last_end']},
         'reqs': STATE['reqs'], 'req_timeouts': STATE['req_timeouts'], 'last_req': STATE['last_req'],
         'max_ms': {'poll': STATE['ms_poll'], 'tick': STATE['ms_tick'], 'accept': STATE['ms_accept'],
                    'frame': STATE['ms_frame'], 'write': STATE['ms_write']}}
    if 'status' in HOOKS:
        try:
            d.update(HOOKS['status']())
        except Exception as e:
            d['status_error'] = repr(e)
    if 'sensors' in HOOKS:
        try:
            d['sensors'] = HOOKS['sensors']()
        except Exception as e:
            d['sensors'] = {'error': repr(e)}
    return d


def setup_page(query):
    secs = query.get('secs', str(CFG.get('setup_secs', 300)))
    return ('<!doctype html><title>%s setup</title>'
            '<meta name=viewport content="width=device-width,initial-scale=1">'
            '<body style="margin:0;background:#000;color:#eee;font:16px sans-serif;text-align:center">'
            '<img id=s src="/stream?secs=%s" style="width:100%%;display:block">'
            '<p id=m>turn the lens until the score peaks</p>'
            '<p id=x style="color:#9a9"></p>'
            '<p><button onclick="act(\'/shoot\')">shoot</button> '
            '<button onclick="act(\'/refresh\')">refresh config</button> '
            '<a style="color:#8cf" href="/setup?secs=%s">restart</a> '
            '<a style="color:#f88" href="/live">live</a></p>'
            '<script>var S="%s",last=-1,same=0;'
            'function re(){document.getElementById("s").src="/stream?secs="+S+"&t="+Date.now();same=0;}'
            'function act(p){fetch(p,{method:"POST"}).then(function(r){return r.text()}).then(function(t){'
            'document.getElementById("x").textContent=p+": "+t;}).catch(function(){});}'
            'setInterval(function(){fetch("/status").then(function(r){return r.json()}).then(function(d){'
            'var m=document.getElementById("m"),u=d.setup;'
            'if(d.mode!=="setup"){m.textContent="session ended ("+u.last_end+") - reconnecting";re();return;}'
            'same=(u.frames===last)?same+1:0;last=u.frames;'
            'var sx="";for(var k in (d.sensors||{})){sx+=" "+k+"="+d.sensors[k];}'
            'm.textContent="score "+u.score+"  best "+u.best+"  "+u.remaining_s+"s  cfg "+d.cfg+(u.viewer?"":" (no viewer)")+sx;'
            'if(same>=2){m.textContent+=" - stream stalled, reconnecting";re();}'
            '}).catch(function(){})},2000);</script>'
            % (secrets.DEVICE, secs, secs, secs))


def _update(conn, headers):
    """Push OTA (developer fast path): same contract as the loader's ota.py."""
    if not OTA_TOKEN or headers.get('x-token') != OTA_TOKEN:
        _reply(conn, '403 Forbidden', 'application/json', '{"error": "bad token"}')
        return False
    n = int(headers.get('content-length', '0'))
    if not 0 < n < 262144:
        _reply(conn, '400 Bad Request', 'application/json', '{"error": "bad length"}')
        return False
    body = b''
    while len(body) < n:
        chunk = conn.read(min(4096, n - len(body)))
        if not chunk:
            break
        body += chunk
    try:
        fw_install(body.decode(), '')       # no prove-out for a pushed build
    except Exception as e:
        _reply(conn, '400 Bad Request', 'application/json', '{"error": "%r"}' % e)
        return False
    _reply(conn, '200 OK', 'application/json', '{"ok": true, "resetting": true}')
    return True


def _dispatch(req, in_session):
    """Handle one request. Returns: a new viewer conn (for /stream while in
    a session), True if a session was run, else None."""
    conn, method, path, query, headers = req
    try:
        if path == '/status':
            _reply(conn, '200 OK', 'application/json', _j(status_dict()))
        elif path in ('/setup', '/'):
            _reply(conn, '200 OK', 'text/html', setup_page(query))
        elif path in ('/live', '/stop'):
            STATE['stop'] = True
            _reply(conn, '200 OK', 'text/html',
                   '<!doctype html><body style="background:#000;color:#eee;font:20px sans-serif;'
                   'text-align:center"><p>%s</p><p><a style="color:#8cf" href="/setup">setup again</a></p>'
                   % ('setup mode ending; recording resumes.' if in_session else 'already live.'))
        elif path == '/stream':
            try:
                secs = int(query.get('secs', CFG.get('setup_secs', 300)))
            except ValueError:
                secs = CFG.get('setup_secs', 300)
            conn.write(b'HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n'
                       b'Cache-Control: no-cache\r\nConnection: close\r\n\r\n')
            if in_session:
                STATE['until'] = max(STATE['until'], time.time() + secs)
                return conn
            setup_session(conn, secs, 'url')
            return True
        elif path == '/shoot' and method == 'POST':
            ok = False
            if 'shoot' in HOOKS:
                try:
                    ok = HOOKS['shoot']()
                except Exception as e:
                    print('shoot error', repr(e))
                if in_session:
                    grayscale_preview()
            _reply(conn, '200 OK', 'text/plain', 'delivered' if ok else 'spooled or failed')
        elif path == '/refresh' and method == 'POST':
            changed = []
            if 'refresh' in HOOKS:
                try:
                    changed = HOOKS['refresh']()
                except Exception as e:
                    print('refresh error', repr(e))
            _reply(conn, '200 OK', 'text/plain', 'cfg %s, changed %s' % (CFG.get('cfg'), changed or 'nothing'))
        elif path == '/update' and method == 'POST':
            if _update(conn, headers):
                conn.close()
                print('control: pushed app installed, resetting')
                time.sleep_ms(300)
                machine.reset()
        else:
            _reply(conn, '404 Not Found', 'application/json', '{"error": "unknown"}')
    except OSError as e:
        print('control: reply error', repr(e))
    conn.close()
    return None


def _timed(key, fn, *a):
    t = time.ticks_ms()
    r = fn(*a)
    d = time.ticks_diff(time.ticks_ms(), t)
    if d > STATE[key]:
        STATE[key] = d
    return r


def setup_session(conn, secs, trigger):
    """Setup mode: stream until timeout / /live / button / viewer gone.
    Blocks the live loop; telemetry via HOOKS['tick'] only while no viewer."""
    STATE['mode'] = 'setup'
    STATE['stop'] = False
    STATE['best'] = STATE['score'] = 0.0
    STATE['until'] = time.time() + secs
    STATE['sessions'] += 1
    for k in ('ms_poll', 'ms_tick', 'ms_accept', 'ms_frame', 'ms_write'):
        STATE[k] = 0
    viewer_lost_at = None if conn else time.time()
    last_refresh = time.time()
    frames = 0
    print('setup: start (%s, %ds)' % (trigger, secs))
    _led_set(True)
    try:
        restore_preview()
        grayscale_preview()
        while True:
            now = time.time()
            if STATE['stop']:
                STATE['last_end'] = 'live'
                break
            if now >= STATE['until']:
                STATE['last_end'] = 'timeout'
                break
            if _btn_edge():
                STATE['last_end'] = 'button'
                break
            if conn is None and trigger == 'url' and viewer_lost_at and now - viewer_lost_at > 30:
                STATE['last_end'] = 'viewer gone'
                break
            if trigger == 'config' and now - last_refresh > 30 and 'refresh' in HOOKS:
                last_refresh = now
                HOOKS['refresh']()
                if CFG.get('mode') != 'setup':
                    STATE['last_end'] = 'config'
                    break
            STATE['viewer'] = conn is not None
            if conn is None and 'tick' in HOOKS:
                _timed('ms_tick', HOOKS['tick'])
            req = _timed('ms_accept', _accept)
            if req:
                new = _dispatch(req, True)
                if new is not None and new is not True:
                    if conn:
                        conn.close()
                    conn = new
                    viewer_lost_at = None
            if conn is None:
                time.sleep_ms(50)
                continue
            try:
                import sensor
                img = _timed('ms_frame', sensor.snapshot)
                score = sharpness(img)
                STATE['score'] = score
                if score > STATE['best']:
                    STATE['best'] = score
                data = annotate(img, score, STATE['best'], int(STATE['until'] - now))
                t = time.ticks_ms()
                conn.write(('--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(data)).encode())
                mv = memoryview(data)
                for i in range(0, len(data), 4096):
                    conn.write(mv[i:i + 4096])
                conn.write(b'\r\n')
                d = time.ticks_diff(time.ticks_ms(), t)
                if d > STATE['ms_write']:
                    STATE['ms_write'] = d
                frames += 1
                STATE['frames'] += 1
            except OSError as e:
                print('setup: viewer dropped', repr(e))
                conn.close()
                conn = None
                viewer_lost_at = time.time()
    except Exception as e:
        STATE['last_end'] = 'error ' + repr(e)
        print('setup: session error', repr(e))
    finally:
        if conn:
            try:
                conn.close()
            except OSError:
                pass
        STATE['mode'] = 'live'
        STATE['viewer'] = False
        _led_set(False)
        try:
            restore_preview()
        except Exception as e:
            print('setup: restore failed', repr(e))
        gc.collect()
    print('setup: end (%s), %d frames, best %.1f' % (STATE['last_end'], frames, STATE['best']))


def control_poll():
    """Call from wherever the live loop waits. Returns True if a setup
    session ran (the caller resets its motion reference and heartbeat)."""
    if _btn_edge():
        print('setup: button -> start')
        setup_session(None, CFG.get('setup_secs', 300), 'button')
        return True
    req = _timed('ms_accept', _accept)
    if req:
        return _dispatch(req, False) is True
    return False
