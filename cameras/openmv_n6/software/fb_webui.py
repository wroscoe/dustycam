#!/usr/bin/env python3
"""Read the OpenMV camera's framebuffer via the official `openmv` protocol
client (same debug protocol the OpenMV IDE uses) and show it in a web UI.

Run with the project venv: .venv/bin/python fb_webui.py
Then open http://localhost:8080

Unlike the earlier stream_server.py approach, the camera-side script doesn't
need to know about streaming at all — this pulls whatever the running script
leaves in the framebuffer. On connect it stops the running script and execs
SCRIPT_FILE (default: red_square.py) so the state is deterministic.
"""

import io
import struct
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

import omv_patches  # noqa: F401  (fixes hangs/exits in the openmv package)
from openmv.camera import Camera
from openmv.exceptions import OMVException, ResyncException

SERIAL_PORT = "/dev/ttyACM0"
HTTP_PORT = 8080
JPEG_QUALITY = 85

latest_jpeg = None
frame_id = 0
fps = 0.0
device_stdout = ""
cond = threading.Condition()

PAGE = b"""<!doctype html>
<title>OpenMV framebuffer</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:95vw;margin-top:1em}
pre{color:#8c8}</style>
<h3>OpenMV framebuffer (IDE debug protocol)</h3>
<img src="/stream">
<pre id="out"></pre>
<script>
setInterval(async () => {
  const r = await fetch('/status');
  document.getElementById('out').textContent = await r.text();
}, 1000);
</script>
"""


FRAME_INTERVAL = 0.033  # cap host reads (~30 fps); each read locks the camera's
                        # stream channel and steals time from its capture loop
STATUS_INTERVAL = 0.5   # poll stdout/status at 2 Hz instead of every iteration
SESSION_MAX_S = 600     # protocol sessions degrade over hours; refresh proactively
STALL_RESET_S = 8       # no frames this long -> device reset


def start_session(reset):
    """Attach to the camera and stream from whatever /flash/main.py is doing.

    Important: never use the protocol's stop()/exec() here. Stopping or
    replacing a running script soft-reboots the device, which resurrects
    flash main.py and clears the streaming flag — an endless tug-of-war.
    (Deploy script changes with: mpremote cp <script> :/flash/main.py)

    reset=False just reconnects (the camera keeps running, no video gap).
    reset=True SYS_RESETs the device first — needed when the stream is wedged.
    """
    if reset:
        cam = Camera(port=SERIAL_PORT, timeout=2.0, ack=False)
        cam.connect()
        cam.reset()  # SYS_RESET; disconnects automatically
        time.sleep(4)

    cam = Camera(port=SERIAL_PORT, timeout=2.0, ack=False)
    cam.connect()
    cam.streaming(True)
    print(f"session started (reset={reset})", flush=True)
    return cam


def reader():
    global latest_jpeg, frame_id, fps, device_stdout
    need_reset = True  # first attach resets to guarantee a clean stream
    while True:
        try:
            cam = start_session(need_reset)
            need_reset = False
            session_t0 = time.monotonic()
            last_status = 0.0
            last_frame_at = time.monotonic()
            errors = 0
            count, t0 = 0, time.monotonic()
            while True:
                now = time.monotonic()
                if now - session_t0 > SESSION_MAX_S:
                    print("periodic session refresh", flush=True)
                    cam.disconnect()
                    break

                try:
                    if now - last_status > STATUS_INTERVAL:
                        last_status = now
                        status = cam.read_status()
                        if status and status.get("stdout"):
                            if text := cam.read_stdout():
                                device_stdout = text.strip().splitlines()[-1]
                    frame = cam.read_frame()
                    errors = 0
                except (OMVException, ResyncException):
                    errors += 1
                    if errors > 10:
                        need_reset = True
                        raise  # give up on this session, reconnect
                    time.sleep(0.1)
                    continue

                if not frame:
                    if now - last_frame_at > STALL_RESET_S:
                        print("stream stalled, resetting device", flush=True)
                        need_reset = True
                        cam.disconnect()
                        break
                    time.sleep(0.005)
                    continue
                last_frame_at = now

                img = Image.frombytes(
                    "RGB", (frame["width"], frame["height"]), frame["data"]
                )
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=JPEG_QUALITY)
                with cond:
                    latest_jpeg = buf.getvalue()
                    frame_id += 1
                    cond.notify_all()

                count += 1
                if count == 30:
                    now = time.monotonic()
                    fps = count / (now - t0)
                    count, t0 = 0, now

                time.sleep(FRAME_INTERVAL)
        except Exception as e:
            print(f"session error ({type(e).__name__}: {e}), reconnecting in 3s",
                  flush=True)
            try:
                cam.disconnect()
            except Exception:
                pass
            time.sleep(3)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/status":
            body = f"host fps: {fps:.1f}   device: {device_stdout}".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/frame.jpg":
            with cond:
                cond.wait_for(lambda: latest_jpeg is not None, timeout=5)
                frame = latest_jpeg
            if frame is None:
                self.send_error(503, "no frame yet")
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
                        frame, last_id = latest_jpeg, frame_id
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


def reader_guard():
    try:
        reader()
    except Exception:
        import traceback

        traceback.print_exc()
        raise


def main():
    t = threading.Thread(target=reader_guard, daemon=True)
    t.start()
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    print(f"serving on http://localhost:{HTTP_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
