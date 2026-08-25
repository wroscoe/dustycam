"""OTA updater: pull logger.py from sensorhub during the hourly wake.

Safety stack (see LESSONS.md in wavesharecam_sandbox for the field history):
  - main.py (the loader) is immutable; only logger.py is ever replaced
  - downloads are compile()-checked, installed by atomic rename, previous
    version kept as logger_prev.py
  - install writes fw_pending; logger calls mark_valid() only after a fully
    successful cycle -- the loader rolls back if that never happens
  - a rolled-back version is blacklisted in fw_bad.txt and never retried
  - no update attempts below MIN_BATT_PCT unless on USB power
"""
import os

import machine
import secrets

HUB = getattr(secrets, "HUB", "http://%s:8088" % secrets.MQTT_HOST)
NAME = "plant"
VERSION_FILE = "fw_version.txt"
PENDING = "fw_pending"
BAD = "fw_bad.txt"
MIN_BATT_PCT = 20


def _read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def http_get(url, timeout_s=15, max_len=131072):
    import socket
    proto, _, host, path = url.split("/", 3)
    path = "/" + path
    port = 80
    if ":" in host:
        host, port = host.split(":")
        port = int(port)
    addr = socket.getaddrinfo(host, port)[0][-1]
    s = socket.socket()
    s.settimeout(timeout_s)
    try:
        s.connect(addr)
        s.send(("GET %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n"
                % (path, host)).encode())
        resp = b""
        while len(resp) < max_len:
            chunk = s.recv(2048)
            if not chunk:
                break
            resp += chunk
    finally:
        s.close()
    head, _, body = resp.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, body


def mark_valid():
    """Called by logger after a fully successful cycle: the pending update
    (if any) is now the trusted version."""
    ver = _read(PENDING)
    if not ver:
        return
    with open(VERSION_FILE, "w") as f:
        f.write(ver)
    for p in (PENDING, PENDING + ".n"):
        try:
            os.remove(p)
        except OSError:
            pass
    print("fw", ver, "marked valid")


def check(batt_pct, vbus, log_err):
    """One cheap version probe per wake; install + reset on a new version.
    Never raises."""
    try:
        if not vbus and batt_pct is not None and batt_pct < MIN_BATT_PCT:
            return
        status, body = http_get(HUB + "/firmware/%s/version" % NAME, 5, 256)
        if status != 200:
            return
        remote = body.decode().strip()
        if not remote or remote == _read(VERSION_FILE) or remote == _read(BAD):
            return
        if _read(PENDING):          # an install is already awaiting prove-out
            return
        print("fw update:", _read(VERSION_FILE) or "?", "->", remote)
        status, code = http_get(HUB + "/firmware/%s.py" % NAME)
        if status != 200 or not code:
            log_err("ota fetch %d" % status)
            return
        compile(code, "logger_new.py", "exec")     # syntax gate
        with open("logger_new.py", "wb") as f:
            f.write(code)
        try:
            os.remove("logger_prev.py")
        except OSError:
            pass
        os.rename("logger.py", "logger_prev.py")
        os.rename("logger_new.py", "logger.py")
        with open(PENDING, "w") as f:              # prove-out starts now
            f.write(remote)
        try:
            os.remove(PENDING + ".n")
        except OSError:
            pass
        print("fw installed, rebooting")
        machine.reset()
    except Exception as e:                         # OTA must never kill a cycle
        log_err("ota: %r" % e)
