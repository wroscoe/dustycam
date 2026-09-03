"""otapull: pull firmware from sensorhub. Standard §4 "Firmware".

Loader contract (ota_main.py, never updated OTA): it runs app.py, and on a
crash restores app_prev.py and reboots. This module adds the prove-out on
top, entirely inside app.py so the loader needs no change:

  install  -> app.py swapped, fw_pending.txt = new version, reset
  boot     -> fw_boot_check(): pending exists and we are NOT that version
              => the loader rolled us back; pending becomes fw_bad.txt
  first successful upload -> fw_mark_valid(): pending removed
  fw_check() never installs a version equal to fw_bad.txt, and never while
  a pending install is still proving itself.
"""
import os

import machine
import secrets

from uplink import *

FW_PENDING = '/flash/fw_pending.txt'
FW_BAD = '/flash/fw_bad.txt'


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ''


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def fw_boot_check(version):
    """Call once at boot. Returns the version that was rolled back (now
    blacklisted), or None."""
    pending = _read(FW_PENDING)
    if pending and pending != version:
        with open(FW_BAD, 'w') as f:
            f.write(pending)
        _rm(FW_PENDING)
        print('fw: %s crashed and was rolled back; blacklisted' % pending)
        return pending
    return None


def fw_pending():
    return _read(FW_PENDING)


def fw_mark_valid():
    """Called after a fully successful upload: the pending version is good."""
    if _read(FW_PENDING):
        _rm(FW_PENDING)
        print('fw: marked valid')


def fw_install(src, remote):
    compile(src, 'app.py', 'exec')          # syntax gate
    with open('/flash/app_new.py', 'w') as f:
        f.write(src)
    _rm('/flash/app_prev.py')
    try:
        os.rename('/flash/app.py', '/flash/app_prev.py')
    except OSError:
        pass
    os.rename('/flash/app_new.py', '/flash/app.py')
    with open(FW_PENDING, 'w') as f:
        f.write(remote)


def fw_check(version):
    """One version probe; install + reset on a new, non-blacklisted version.
    Never raises. Returns True if an install happened (the board resets)."""
    try:
        if _read(FW_PENDING):
            return False
        status, body = http_get('/firmware/%s/version' % secrets.DEVICE, 256, 8)
        if status != 200:
            return False
        remote = body.decode().strip()
        if not remote or remote == version or remote == _read(FW_BAD):
            return False
        print('fw update: %s -> %s' % (version, remote))
        status, code = http_get('/firmware/%s.py' % secrets.DEVICE, 262144, 40)
        if status != 200 or not code:
            print('fw fetch failed', status)
            return False
        fw_install(code.decode(), remote)
        print('fw installed, resetting')
        import time
        time.sleep_ms(300)
        machine.reset()
        return True
    except Exception as e:
        print('fw check error:', repr(e))
        return False
