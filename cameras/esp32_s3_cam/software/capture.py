#!/usr/bin/env python3
"""Capture a VGA photo from the Waveshare ESP32-S3-CAM (GC0308, MicroPython
camera firmware) over USB serial and save it as a PNG.

Usage: python3 capture.py [output.png]

Requires only the Python stdlib + pyserial. See LESSONS.md for the full
board setup story. The board must be running the micropython-camera-API
ESP32_GENERIC_S3-SPIRAM_OCT firmware and enumerate as 303a:4001.
"""
import serial, time, base64, zlib, struct, sys

PORT = '/dev/ttyACM0'
W, H = 640, 480
CHUNK = 4096

# Waveshare ESP32-S3-CAM pin map (NOT the real ESP32-S3-EYE map)
DEVICE_CODE = (
    "from camera import Camera, PixelFormat, FrameSize\n"
    "import binascii\n"
    "cam = Camera(data_pins=[45,47,48,46,42,40,39,21],"
    "vsync_pin=17,href_pin=18,sda_pin=8,scl_pin=7,pclk_pin=41,xclk_pin=38,"
    "xclk_freq=20000000,pixel_format=PixelFormat.RGB565,"
    "frame_size=FrameSize.VGA,init=False)\n"
    "cam.init()\n"
    "for _ in range(5):\n"
    "    img = cam.capture()\n"   # warm-up frames for auto-exposure
    "print('LEN', len(img))\n"
    "mv = memoryview(img)\n"
    f"CH = {CHUNK}\n"
    "for i in range(0, len(img), CH):\n"
    "    print(str(i) + ':' + binascii.b2a_base64(mv[i:i+CH]).decode().strip())\n"
    "print('EOF_MARKER')\n"
    "cam.deinit()\n"
).encode()


def open_port():
    s = serial.Serial()
    s.port = PORT; s.baudrate = 115200; s.timeout = 0.5
    s.dtr = False; s.rts = False   # MUST pre-set before open() or the chip resets
    s.open()
    return s


def drain(s, t=0.5):
    end = time.time() + t; b = b''
    while time.time() < end:
        d = s.read(65536)
        if d: b += d; end = time.time() + 0.3
    return b


def capture_frame():
    s = open_port()
    # Soft-reboot first: raw REPL is unreliable on the first session after
    # boot, and re-initializing the camera in a dirty session hard-hangs it.
    s.write(b'\r\x03\x03'); drain(s, 0.8)
    s.write(b'\x02'); drain(s, 0.5)
    s.write(b'\x04'); time.sleep(2.0); drain(s, 1.0)
    s.write(b'\r\x03\x03'); drain(s, 0.8)
    s.write(b'\x01'); time.sleep(0.4); drain(s, 0.6)   # raw REPL

    # feed code in small pieces (no flow control on the CDC link)
    for i in range(0, len(DEVICE_CODE), 128):
        s.write(DEVICE_CODE[i:i+128]); time.sleep(0.04)
    time.sleep(0.3)
    s.write(b'\x04')
    ack = s.read(2)
    if ack != b'OK':
        raise RuntimeError(f"raw REPL exec not acknowledged: {ack!r}")

    end = time.time() + 240; buf = b''
    while time.time() < end:
        try:
            d = s.read(65536)
        except serial.SerialException:
            time.sleep(0.3); continue
        if d:
            buf += d
            if b'EOF_MARKER' in buf: break
    s.close()

    size = None; chunks = {}
    for l in (x.strip() for x in buf.decode('utf-8', 'replace').split('\n')):
        if l.startswith('OK'): l = l[2:]
        if l.startswith('LEN '):
            size = int(l.split()[1]); continue
        if l == 'EOF_MARKER': break
        if ':' in l:
            off_s, b64 = l.split(':', 1)
            try:
                chunks[int(off_s)] = base64.b64decode(b64)
            except Exception:
                pass
    if size is None and chunks:
        size = W * H * 2      # LEN header lost in transit; frame size is known
    if not size or not chunks:
        raise RuntimeError("no frame received; tail: " + buf[-300:].decode('utf-8', 'replace'))
    frame = bytearray(size)
    for off, d in chunks.items():
        frame[off:off+len(d)] = d
    missing = sorted(set(range(0, size, CHUNK)) - set(chunks))
    return bytes(frame), missing


def rgb565_to_png(raw, path):
    rows = bytearray()
    for y in range(H):
        rows.append(0)
        base = y * W * 2
        for x in range(W):
            i = base + x * 2
            v = (raw[i] << 8) | raw[i+1]        # esp32-camera RGB565 is big-endian
            r = (v >> 11) & 0x1F; g = (v >> 5) & 0x3F; b = v & 0x1F
            rows += bytes(((r*255)//31, (g*255)//63, (b*255)//31))
    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(bytes(rows), 6))
    png += chunk(b'IEND', b'')
    open(path, 'wb').write(png)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'capture.png'
    last_err = None
    for attempt in range(3):   # first session after a (re)plug often fails
        try:
            frame, missing = capture_frame()
            break
        except RuntimeError as e:
            last_err = e
            print(f"attempt {attempt+1} failed ({str(e)[:80]}...), retrying")
            time.sleep(1)
    else:
        raise SystemExit(f"all attempts failed: {last_err}")
    rgb565_to_png(frame, out)
    note = f" ({len(missing)} lost 4K chunks -> black stripes)" if missing else ""
    print(f"saved {out}{note}")
