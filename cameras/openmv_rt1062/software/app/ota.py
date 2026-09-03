"""Tiny push-OTA listener for /flash/app.py. Deployed to /flash/ota.py.

Stable module — never updated over OTA (only main.py and this file are
outside the OTA loop). The app calls ota.poll() from its idle loop, so a
push lands within ~100 ms even while the camera loop runs; main.py keeps
polling in recovery mode when app.py is broken, so a bad push can always
be fixed with another push.

Protocol (LAN-only, shared-secret header):
  GET  /status               -> {"version": ..., "uptime_s": ...}
  POST /update               -> body = new app.py source
       X-Token: <secrets.OTA_TOKEN>

Install path: compile-check the source (syntax error -> 400, nothing
touched), then app.py -> app_prev.py, new -> app.py, respond 200, reset.
A runtime crash after that is rolled back by main.py (app_prev.py
restored, broken file kept as app_bad.py for debugging).
"""
import os
import socket
import sys
import time

import machine
import network

state = {'version': '?'}
_srv = None
_token = ''
_t0 = time.ticks_ms()


def wifi_connect(ssid, password, timeout_s=15):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return wlan
    try:
        wlan.connect(ssid, password)
    except OSError:
        return None
    end = time.time() + timeout_s
    while time.time() < end:
        if wlan.isconnected():
            return wlan
        time.sleep_ms(300)
    return None


def start(port, token):
    global _srv, _token
    _token = token
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', port))
    s.listen(1)
    s.setblocking(False)
    _srv = s
    print('ota: listening on :%d' % port)


def _send(conn, code, body):
    conn.write(('HTTP/1.1 %s\r\nContent-Type: application/json\r\n'
                'Content-Length: %d\r\nConnection: close\r\n\r\n'
                % (code, len(body))).encode())
    conn.write(body.encode())


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _install(src):
    compile(src, 'app.py', 'exec')          # syntax gate; raises on bad source
    with open('/flash/app_new.py', 'w') as f:
        f.write(src)
    _rm('/flash/app_prev.py')
    try:
        os.rename('/flash/app.py', '/flash/app_prev.py')
    except OSError:
        pass                                # no current app.py (recovery)
    os.rename('/flash/app_new.py', '/flash/app.py')


def _handle(conn):
    """Returns True if the device should reset."""
    line = conn.readline()
    if not line:
        return False
    parts = line.split()
    method, path = parts[0], parts[1]
    headers = {}
    while True:
        l = conn.readline()
        if not l or l == b'\r\n':
            break
        k, _, v = l.decode().partition(':')
        headers[k.strip().lower()] = v.strip()

    if method == b'GET' and path == b'/status':
        _send(conn, '200 OK', '{"version": "%s", "uptime_s": %d}'
              % (state['version'], time.ticks_diff(time.ticks_ms(), _t0) // 1000))
        return False

    if method == b'POST' and path == b'/update':
        if not _token or headers.get('x-token') != _token:
            _send(conn, '403 Forbidden', '{"error": "bad token"}')
            return False
        n = int(headers.get('content-length', '0'))
        if not 0 < n < 131072:
            _send(conn, '400 Bad Request', '{"error": "bad length"}')
            return False
        body = b''
        while len(body) < n:
            chunk = conn.read(min(4096, n - len(body)))
            if not chunk:
                break
            body += chunk
        try:
            _install(body.decode())
        except Exception as e:
            _send(conn, '400 Bad Request', '{"error": "%r"}' % e)
            return False
        _send(conn, '200 OK', '{"ok": true, "resetting": true}')
        return True

    _send(conn, '404 Not Found', '{"error": "unknown"}')
    return False


def poll():
    """Non-blocking; call from the app idle loop. May reset the device."""
    if _srv is None:
        return
    try:
        conn, _ = _srv.accept()
    except OSError:
        return
    reset = False
    try:
        conn.setblocking(True)
        conn.settimeout(8)
        reset = _handle(conn)
    except Exception as e:
        sys.print_exception(e)
    finally:
        conn.close()
    if reset:
        print('ota: installed new app.py, resetting')
        time.sleep_ms(300)
        machine.reset()
