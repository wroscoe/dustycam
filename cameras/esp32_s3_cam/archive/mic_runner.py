#!/usr/bin/env python3
"""Run the ES7210 mic level test on the Waveshare ESP32-S3-CAM over raw
serial, using the DTR/RTS-safe protocol from wavesharecam_sandbox/capture.py
(mpremote wedges this board's USB-CDC link)."""
import serial
import time
import sys

PORT = '/dev/ttyACM0'

DEVICE_CODE = open(sys.argv[1], 'rb').read() if len(sys.argv) > 1 else None
if DEVICE_CODE is None:
    print("usage: mic_runner.py <device_script.py>")
    sys.exit(1)


def open_port():
    s = serial.Serial()
    s.port = PORT
    s.baudrate = 115200
    s.timeout = 0.5
    s.dtr = False
    s.rts = False   # MUST pre-set before open() or the chip resets
    s.open()
    return s


def drain(s, t=0.5):
    end = time.time() + t
    b = b''
    while time.time() < end:
        d = s.read(65536)
        if d:
            b += d
            end = time.time() + 0.3
    return b


s = open_port()
# soft-reboot first; raw REPL is unreliable on a dirty session
s.write(b'\r\x03\x03'); drain(s, 0.8)
s.write(b'\x02'); drain(s, 0.5)
s.write(b'\x04'); time.sleep(2.0); drain(s, 1.0)
s.write(b'\r\x03\x03'); drain(s, 0.8)
s.write(b'\x01'); time.sleep(0.4); drain(s, 0.6)   # raw REPL

for i in range(0, len(DEVICE_CODE), 128):
    s.write(DEVICE_CODE[i:i + 128])
    time.sleep(0.04)
time.sleep(0.3)
s.write(b'\x04')
ack = s.read(2)
if ack != b'OK':
    raise RuntimeError(f"raw REPL exec not acknowledged: {ack!r}")

end = time.time() + 75
buf = b''
while time.time() < end:
    try:
        d = s.read(65536)
    except serial.SerialException:
        time.sleep(0.3)
        continue
    if d:
        buf += d
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            txt = line.decode(errors='replace').strip()
            if txt and txt != '\x04':
                print(txt, flush=True)
            if 'done' in txt or 'Traceback' in txt:
                end = min(end, time.time() + 3)
s.close()
print("--- runner finished ---")
