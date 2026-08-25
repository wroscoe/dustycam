"""Immutable boot loader -- shipped over USB once, NEVER updated OTA.

Boots logger.py and owns crash rollback for OTA updates (see ota.py):
if an installed update has not proven itself (fw_pending flag) and this
boot is a crash rather than a deep-sleep wake, count it; on the 2nd
crash restore logger_prev.py and blacklist the bad version.
"""
import os
import time

import machine

PENDING = "fw_pending"      # written by ota.install(), cleared by a good cycle
PREV = "logger_prev.py"
BAD = "fw_bad.txt"


def _exists(p):
    try:
        os.stat(p)
        return True
    except OSError:
        return False


def _rollback():
    try:
        ver = open(PENDING).read().strip()
        os.rename(PREV, "logger.py")
        with open(BAD, "w") as f:            # never retry this version
            f.write(ver)
        os.remove(PENDING)
        print("OTA rollback; blacklisted", ver)
    except OSError as e:
        print("rollback failed:", e)


if _exists(PENDING) and machine.reset_cause() != machine.DEEPSLEEP_RESET:
    # a pending update + a non-deepsleep boot = the new logger likely crashed
    try:
        n = int(open(PENDING + ".n").read())
    except (OSError, ValueError):
        n = 0
    n += 1
    with open(PENDING + ".n", "w") as f:
        f.write(str(n))
    if n >= 2 and _exists(PREV):
        _rollback()

# short grace period so a plugged-in USB session can Ctrl-C before work starts
time.sleep(2)

try:
    import logger
    logger.main()
except Exception as e:
    try:
        with open("errors.log", "a") as f:
            f.write("0 boot: %r\n" % e)
    except OSError:
        pass
    time.sleep(30)          # don't tight-loop the watchdog/battery
    machine.reset()
