# OpenMV sandbox

Experiments with an **OpenMV N6** camera (STM32N657, OpenMV firmware 5.0.0,
MicroPython v1.28) connected over USB at `/dev/ttyACM0`.

## Files

| File | Purpose |
|------|---------|
| `fb_webui.py` | **Main viewer.** Reads the camera framebuffer via the official protocol and serves MJPEG at http://localhost:8080. Run: `.venv/bin/python fb_webui.py` |
| `omv_patches.py` | Required monkeypatches for the `openmv` PyPI package (see bugs below). Import before creating `Camera`. |
| `red_square.py` | Camera script: live video + filled red square, prints FPS. Deployed as `/flash/main.py`. |
| `red_square_quiet.py` | Same without the FPS prints. |
| `camera_stream.py` + `stream_server.py` | Alternative, simpler pipeline: camera-side USB-VCP request/response JPEG streamer + host MJPEG server (no protocol library needed, but requires deploying `camera_stream.py` as `main.py`). |
| `sensorhub_cam.py` | Camera app (**`/flash/app.py`** on the second N6, serial `004C347E3643`): motion-gated sensorhub uploader with MQTT telemetry (60 s) and SD offline buffering (frames queue to `/sdcard/pending/` when the server is unreachable; backlog drains oldest-first on reconnect with recording paused; at-least-once → dedupe on meta `ts`). Diffs each frame against the last *uploaded* frame (fraction of pixels with LAB-lightness delta > `DIFF_L_THRESH`); POSTs a VGA JPEG to `http://192.168.86.26:8088/blob/n6cam/frame` only when > `DIFF_MIN_FRAC` changed or every `HEARTBEAT_S`. Meta carries ts/diff/version/ip. Raw-socket POST — the frozen `requests` lib fails (`ValueError: invalid syntax for integer with base 10`) against sensorhub's HTTP/1.0 ingest. **Update via `./ota_push.py`, not USB.** |
| `ota_main.py` | OTA bootstrap (**`/flash/main.py`**, never updated OTA): WiFi → NTP → start OTA listener → `app.run(ota.poll)`. On app crash: revert `app_prev.py` → `app.py` (bad file kept as `app_bad.py`) and reboot; recovery mode (polls OTA forever) if nothing to revert. |
| `ota.py` | OTA listener (**`/flash/ota.py`**, stable): non-blocking HTTP on port 8266, polled from the app idle loop. `GET /status`; `POST /update` (X-Token auth) compile-checks, rotates app.py→app_prev.py, resets. |
| `sensorhub_cam_lp.py` | **Low-power variant** (push as app.py with `./ota_push.py sensorhub_cam_lp.py --wait`): sensor at 5 fps, motion check every 60 s, WiFi off between deliveries (15 s OTA linger window after each; heartbeat guarantees one every 5 min), CPU clock lowered via `cpufreq` if supported, telemetry batched per window. **No live stream in this mode.** Supervise its first boot near USB. |
| `ota_push.py` | Host tool: `./ota_push.py [file] [--ip …] [--wait [MIN]]` pushes a new app.py over WiFi. Auto-discovers the camera IP from the `ip` field in sensorhub blob meta, then polls `/status` until the new version is live. `--wait` retries until a low-power WiFi window opens. |
| `secrets.py` | WiFi + sensorhub + OTA token/tuning (copy `secrets_example.py`; gitignored). Deployed to `/flash/secrets.py` — **not** OTA-managed, so after editing it `mpremote cp` it to `/flash` and reset or the app runs stale config. |
| `main_factory_backup.py` | The factory `main.py` (LED blinker) that shipped on the camera. |
| `omv_protocol.md` | Official OpenMV Protocol V2 spec (from openmv/openmv `docs/protocol.md`). |
| `.venv` | uv venv (Python 3.12) with `openmv==1.0.7`, `pillow`, `numpy`, `pyserial`. |

## Hard-won learnings

Moved to the sargineer warehouse (`http://localhost:8093`) — query it before
touching this board:

```sh
curl -s "http://localhost:8093/notes?hw=openmv-cam-n6"
```
