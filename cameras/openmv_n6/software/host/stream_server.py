#!/usr/bin/env python3
"""Read JPEG frames from an OpenMV cam over USB serial and serve them as MJPEG.

The camera must be running camera_stream.py (deployed as /flash/main.py).
Open http://localhost:8080 to watch, or /frame.jpg for a single frame.
"""

import struct
import threading
import time

import serial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERIAL_PORT = "/dev/ttyACM0"
HTTP_PORT = 8080
MAX_FRAME_SIZE = 5_000_000

latest_frame = None
frame_id = 0
fps = 0.0
cond = threading.Condition()

PAGE = b"""<!doctype html>
<title>OpenMV stream</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:95vw;image-rendering:pixelated;margin-top:1em}</style>
<h3>OpenMV live stream</h3>
<img src="/stream">
"""


def reader():
    """Continuously request frames from the camera, reconnecting on errors."""
    global latest_frame, frame_id, fps
    while True:
        try:
            with serial.Serial(SERIAL_PORT, baudrate=115200, timeout=2) as port:
                port.setDTR(True)
                port.reset_input_buffer()
                count, t0 = 0, time.monotonic()
                while True:
                    port.write(b"snap")
                    port.flush()
                    hdr = port.read(4)
                    if len(hdr) != 4:
                        port.reset_input_buffer()
                        continue
                    (size,) = struct.unpack("<L", hdr)
                    if not 0 < size < MAX_FRAME_SIZE:
                        port.reset_input_buffer()
                        continue
                    buf = port.read(size)
                    if len(buf) != size or not buf.startswith(b"\xff\xd8"):
                        port.reset_input_buffer()
                        continue
                    with cond:
                        latest_frame = buf
                        frame_id += 1
                        cond.notify_all()
                    count += 1
                    if count == 30:
                        now = time.monotonic()
                        fps = count / (now - t0)
                        count, t0 = 0, now
        except serial.SerialException as e:
            print(f"serial error ({e}), retrying in 1s")
            time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/frame.jpg":
            with cond:
                cond.wait_for(lambda: latest_frame is not None, timeout=5)
                frame = latest_frame
            if frame is None:
                self.send_error(503, "no frame from camera yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            last_id = 0
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: frame_id != last_id, timeout=5)
                        if frame_id == last_id:
                            continue
                        frame, last_id = latest_frame, frame_id
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    threading.Thread(target=reader, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    print(f"serving on http://localhost:{HTTP_PORT}")

    def report():
        while True:
            time.sleep(10)
            print(f"camera fps: {fps:.1f}")

    threading.Thread(target=report, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
