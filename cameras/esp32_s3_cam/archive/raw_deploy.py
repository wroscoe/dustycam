#!/usr/bin/env python3
"""Deploy via raw fd + termios only — no pyserial, no DTR/RTS ioctls
(this board's CDC wedges on modem-control transfers while streaming).
Feeds deploy_out.py through MicroPython raw REPL in 128 B chunks."""
import os, sys, time, select, termios

PORT = "/dev/serial/by-id/usb-Espressif_Systems_Espressif_Device_a4cb8fd781900000-if00"
code = open(sys.argv[1], "rb").read()

fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
attrs = termios.tcgetattr(fd)
attrs[0] = attrs[1] = attrs[3] = 0
attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
termios.tcsetattr(fd, termios.TCSANOW, attrs)

def rd(sec, until=None):
    out = b""; t0 = time.time()
    while time.time() - t0 < sec:
        r, _, _ = select.select([fd], [], [], 0.2)
        if r:
            try: out += os.read(fd, 4096)
            except BlockingIOError: pass
        if until and until in out:
            break
    return out

os.write(fd, b"\x03\x03")
rd(1)
os.write(fd, b"\r\x01")                       # enter raw REPL
r = rd(3, b"raw REPL; CTRL-B to exit")
assert b"raw REPL" in r, "no raw REPL: %r" % r[-120:]
rd(0.5)

for i in range(0, len(code), 128):
    os.write(fd, code[i:i+128])
    time.sleep(0.01)
os.write(fd, b"\x04")                          # execute
out = rd(20, b"\x04")
print("exec response:", out[:200].decode("utf-8", "replace"))
if b"Traceback" in out:
    print("BOARD ERROR:\n", out.decode("utf-8", "replace"))
    sys.exit(1)
print("deployed; board should be resetting now")
os.close(fd)
