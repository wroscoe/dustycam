#!/usr/bin/env python3
"""Run app.run() on the board and stream its output — loss-tolerant.

Handles this board's quirks (see LESSONS.md):
- naive port open resets the chip (pre-set DTR/RTS)
- first raw-REPL session after boot corrupts code (soft-reboot + retry)
- CDC drops ~2% of output bytes (idle-timeout exit + APP_DONE sentinel)

Usage: run_app.py [iterations]
"""
import sys
import time

import serial

import glob
PORT = (sorted(glob.glob('/dev/serial/by-id/usb-Espressif*')) or ['/dev/ttyACM0'])[0]
IDLE_EXIT_S = 10
ATTEMPTS = 3


def open_port():
    s = serial.Serial()
    s.port = PORT; s.baudrate = 115200; s.timeout = 0.3
    s.dtr = False; s.rts = False
    s.open()
    return s


def drain(s, t=0.5):
    end = time.time() + t
    while time.time() < end:
        if s.read(65536):
            end = time.time() + 0.3


def attempt(expr):
    """One full session. Returns (success, produced_output)."""
    s = open_port()
    # soft-reboot first: raw REPL is only reliable on a freshly rebooted session
    s.write(b'\r\x03\x03'); drain(s, 0.8)
    s.write(b'\x02'); drain(s, 0.5)
    s.write(b'\x04'); time.sleep(2.0); drain(s, 1.0)
    s.write(b'\r\x03\x03'); drain(s, 0.8)
    s.write(b'\x01'); time.sleep(0.4); drain(s, 0.6)

    code = (f"try:\n"
            f"    {expr}\n"
            f"finally:\n"
            f"    print('APP_DONE')\n").encode()
    for i in range(0, len(code), 128):
        s.write(code[i:i+128]); time.sleep(0.04)
    time.sleep(0.3)
    s.write(b'\x04')
    if s.read(2) != b'OK':
        s.close()
        return False, False

    last_data = time.time()
    pending = b''
    done = False
    produced = False
    failed = False
    while not done and time.time() - last_data < IDLE_EXIT_S:
        try:
            d = s.read(4096)
        except serial.SerialException:
            time.sleep(0.3); continue
        if not d:
            continue
        last_data = time.time()
        pending += d
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            text = line.decode('utf-8', 'replace').strip().lstrip('\x04>')
            if 'APP_DONE' in text:
                done = True; break
            if 'SyntaxError' in text or 'Traceback' in text:
                failed = True
            if text and not failed:
                print(text, flush=True)
                produced = True
    s.close()
    return (done or produced) and not failed, produced


def main():
    # arg is either an iteration count (legacy) or a full expression
    arg = sys.argv[1] if len(sys.argv) > 1 else '10'
    expr = f"import app; app.run({arg})" if arg.isdigit() else arg
    for n in range(1, ATTEMPTS + 1):
        ok, _ = attempt(expr)
        if ok:
            return 0
        print(f"[run_app] attempt {n} failed, retrying...", file=sys.stderr)
        time.sleep(2)
    print("[run_app] giving up — try 'make reset' or replug", file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
