#!/usr/bin/env python3
"""Push a new app.py to the N6 over WiFi.

Usage: ./ota_push.py [file] [--ip CAMERA_IP]

Default file is sensorhub_cam.py. The camera IP is auto-discovered from the
"ip" field the app reports in its sensorhub upload metadata; pass --ip for a
camera that has never uploaded (e.g. sitting in recovery mode after a WiFi
change). Token/port come from ~/.dusty/ (or the generated secrets.py beside this script).

After the push the camera compile-checks, swaps app.py (keeping the old one
as app_prev.py for automatic rollback), and reboots; this script then polls
/status until it reports the new version.
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DB = Path.home() / 'code/sensorhub/data/sensorhub.db'


def load_secrets():
    """OTA settings, from ~/.dusty/ when available.

    Falls back to the generated secrets.py next to this script so the tool
    still runs standalone (no install, or from a checkout without the
    `dusty` package importable).
    """
    try:
        from dusty.generate import TARGETS, flatten
        from dusty.config import load
        cfg = load(camera='openmv_n6')
        return flatten(cfg, TARGETS['openmv_n6']['sections'])
    except Exception:
        ns = {}
        exec((HERE / 'secrets.py').read_text(), ns)
        return ns


def discover_ip(device):
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    rows = con.execute(
        "SELECT meta FROM blobs WHERE device=? ORDER BY ts DESC LIMIT 20", (device,))
    for (meta,) in rows:
        try:
            ip = json.loads(meta).get('ip')
            if ip:
                return ip
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


def status(ip, port, timeout=3):
    with urllib.request.urlopen(f'http://{ip}:{port}/status', timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file', nargs='?', default='sensorhub_cam.py')
    ap.add_argument('--ip', help='camera IP (default: discover from sensorhub DB)')
    ap.add_argument('--wait', nargs='?', const=7, type=float, metavar='MIN',
                    help='retry until the camera is reachable, up to MIN minutes '
                         '(default 7). Needed for the low-power app, whose WiFi '
                         'is only up ~15 s after each delivery/heartbeat.')
    args = ap.parse_args()

    sec = load_secrets()
    port, token, device = sec['OTA_PORT'], sec['OTA_TOKEN'], sec['DEVICE']

    ip = args.ip or discover_ip(device)
    if not ip:
        sys.exit(f'no ip in recent {device} uploads — pass --ip')

    src = (HERE / args.file).read_bytes()
    deadline = time.time() + (args.wait or 0) * 60
    while True:
        try:
            print(f'camera {ip}: currently {status(ip, port)}')
            break
        except OSError as e:
            if time.time() < deadline:
                print('.', end='', flush=True)
                time.sleep(3)
                continue
            sys.exit(f'camera {ip}:{port} unreachable: {e}'
                     + ('' if args.wait else ' (low-power app? try --wait)'))

    req = urllib.request.Request(
        f'http://{ip}:{port}/update', data=src, method='POST',
        headers={'X-Token': token, 'Content-Type': 'text/x-python'})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print('push:', r.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f'push rejected: {e.code} {e.read().decode()}')

    print('waiting for reboot', end='', flush=True)
    # low-power app only opens a WiFi window after its first delivery, so
    # allow up to ~7 min before declaring failure
    for _ in range(210):
        time.sleep(2)
        print('.', end='', flush=True)
        try:
            s = status(ip, port)
            print(f'\nback up: {s}')
            return
        except OSError:
            continue
    sys.exit('\ncamera did not come back — check sensorhub for a new ip, or USB')


if __name__ == '__main__':
    main()
