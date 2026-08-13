"""Autostart shim with SAFE MODE — the only main.py allowed on this board.

Boot order:
1. Wait BOOT_GRACE_S seconds. If the BOOT button (GPIO0) is pressed at any
   point in that window, or /disable.txt exists, DO NOT start the app —
   fall through to the REPL. This is the escape hatch lesson #17 demands.
2. Otherwise run the logger under a watchdog, with echo=False (no prints —
   printing with no host attached wedges the board).

Recovery ladder if this ever wedges anyway: BOOT-held replug, then
`esptool erase-region 0x200000 0x100000` (wipes filesystem, firmware intact).
"""
import time

import machine

BOOT_GRACE_S = 6


def _safe_mode_requested():
    try:
        import os
        os.stat('/disable.txt')
        return True
    except OSError:
        pass
    btn = machine.Pin(0, machine.Pin.IN, machine.Pin.PULL_UP)
    end = time.time() + BOOT_GRACE_S
    while time.time() < end:
        if btn.value() == 0:      # BOOT pressed
            return True
        time.sleep(0.1)
    return False


if not _safe_mode_requested():
    # 2 min: must outlast one capture+encode+save+upload round. Fed by the
    # logger loop itself, so ANY stall (camera, flash, WiFi) reboots us.
    wdt = machine.WDT(timeout=120_000)
    import logger_app
    logger_app.run(echo=False, wdt=wdt)
