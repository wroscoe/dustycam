#!/usr/bin/env python3
"""Watch the board's serial console (USB-Serial/JTAG) and print lines.

Used with the person-detection IDF firmware, whose scores stream on the
console. DTR/RTS-safe open; reconnects if the port hiccups.

Usage: monitor.py [seconds]   (default 30)
"""
import sys
import time

import serial

import glob
PORT = (sorted(glob.glob('/dev/serial/by-id/usb-Espressif*')) or ['/dev/ttyACM0'])[0]


def open_port():
    s = serial.Serial()
    s.port = PORT; s.baudrate = 115200; s.timeout = 0.3
    s.dtr = False; s.rts = False
    s.open()
    return s


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    end = time.time() + duration
    s = None
    pending = b''
    while time.time() < end:
        if s is None:
            try:
                s = open_port()
            except Exception:
                time.sleep(0.5); continue
        try:
            d = s.read(4096)
        except serial.SerialException:
            try: s.close()
            except Exception: pass
            s = None
            time.sleep(0.5); continue
        if not d:
            continue
        pending += d
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            text = line.decode('utf-8', 'replace').rstrip()
            if text:
                print(text, flush=True)
    if s:
        s.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
