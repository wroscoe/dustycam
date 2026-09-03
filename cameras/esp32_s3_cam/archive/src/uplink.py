"""WiFi + HTTP upload of pending images to the local collection server.

Credentials/server in secrets.py (copy secrets_example.py; never committed).
"""
import socket
import time

import network
import storage

try:
    import secrets
except ImportError:
    secrets = None


def wifi_connect(timeout_s=15):
    """Try to join WiFi. Returns True if connected."""
    if secrets is None:
        return False
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return True
    try:
        wlan.connect(secrets.WIFI_SSID, secrets.WIFI_PASS)
    except OSError:
        return False
    end = time.time() + timeout_s
    while time.time() < end:
        if wlan.isconnected():
            return True
        time.sleep(0.5)
    return False


def _post_file(host, port, path, fname, data):
    """Minimal HTTP POST. Returns True on 2xx."""
    s = socket.socket()
    s.settimeout(10)
    try:
        s.connect(socket.getaddrinfo(host, port)[0][-1])
        head = ('POST %s HTTP/1.1\r\n'
                'Host: %s\r\n'
                'X-Filename: %s\r\n'
                'Content-Type: image/jpeg\r\n'
                'Content-Length: %d\r\n'
                'Connection: close\r\n\r\n' % (path, host, fname, len(data)))
        s.write(head.encode())
        mv = memoryview(data)
        for i in range(0, len(data), 1024):
            s.write(mv[i:i + 1024])
        resp = s.read(15)
        return b' 2' in resp[:13]
    except OSError:
        return False
    finally:
        s.close()


def upload_pending(max_files=20):
    """Upload up to max_files pending images. Returns count uploaded."""
    if secrets is None or not wifi_connect():
        return 0
    host, port = secrets.SERVER_HOST, secrets.SERVER_PORT
    n = 0
    for p in storage.pending()[:max_files]:
        try:
            with open(p, 'rb') as f:
                data = f.read()
        except OSError:
            continue
        fname = p.rsplit('/', 1)[1]
        if _post_file(host, port, '/upload', fname, data):
            storage.mark_sent(p)
            n += 1
        else:
            break     # server unreachable; retry next round
    return n
