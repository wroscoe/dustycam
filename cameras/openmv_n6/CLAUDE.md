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
| `secrets.py` | WiFi + sensorhub + OTA token/tuning. **Generated** from `~/.dusty/` by `dusty generate openmv_n6` — edit `~/.dusty/secrets.toml`, not this file. Deployed to `/flash/secrets.py`. |
| `main_factory_backup.py` | The factory `main.py` (LED blinker) that shipped on the camera. |
| `omv_protocol.md` | Official OpenMV Protocol V2 spec (from openmv/openmv `docs/protocol.md`). |
| `.venv` | uv venv (Python 3.12) with `openmv==1.0.7`, `pillow`, `numpy`, `pyserial`. |

## Hard-won learnings (firmware 5.0, Aug 2026)

### The old IDE debug protocol is gone
Firmware 5.0 replaced the classic `usbdbg`/`pyopenmv.py` protocol with a
channel-based **OpenMV Protocol V2** (stdin=1, stdout=2, stream=3). The
official host client is the `openmv` package on PyPI (repo:
`openmv/openmv-python`), which also ships an `openmv` CLI with a pygame viewer.

### `openmv==1.0.7` library bugs (patched in `omv_patches.py`)
- `Transport.recv_packet` resets its timeout clock whenever an *event* packet
  arrives. While streaming, the camera emits frame-ready/stdout events
  continuously (~50-120/s), so a single lost command response makes the call
  spin forever. The patch stops events from resetting the clock; the library's
  resync/retry then works.
- `Camera._send_cmd_wait_resp` has an `except Exception: sys.exit(0)` that
  silently kills the host process on e.g. a serial glitch. Patch re-raises.
- Use `Camera(port, timeout=2.0, ack=False)`. The official CLI defaults to
  `ack=False`; the library constructor's `ack=True` default is not what the
  vendor actually tests with.

### Never `stop()`/`exec()` a running script over the protocol
Stopping (or replacing) a running script triggers a **soft reboot**, which
auto-restarts `/flash/main.py` and clears the streaming flag → an endless
tug-of-war where `read_frame()` yields nothing and sessions wedge.

**Stable recipe** (what `fb_webui.py` does):
1. Deploy the camera script as flash main.py:
   `mpremote cp red_square.py :/flash/main.py`
2. Protocol-connect → `cam.reset()` (SYS_RESET) → wait ~4 s → reconnect.
3. `cam.streaming(True)` → poll `cam.read_frame()` (returns RGB888 dict).
The stream shows whatever the running script's framebuffer holds — drawings
included, no `img.flush()` needed.

### Firmware 5.0 MicroPython API break
`img.draw_rectangle(x, y, w, h, ...)` now raises
`TypeError: object 'int' isn't a tuple or list`. Pass a tuple:
`img.draw_rectangle((x, y, w, h), color=(255, 0, 0), fill=True)`.
A crashing `main.py` **crash-loops silently** (crash → soft reboot → crash …).
Symptom: streaming "works" but script effects (drawings, pixformat changes)
never appear, and stdout shows `>>>` REPL prompts.

### Debugging camera scripts
- MicroPython tracebacks appear on the protocol **stdout channel**:
  `cam.read_status()` / `cam.read_stdout()`. Read it first when behavior is odd.
- `mpremote` (raw REPL) and the protocol share the same USB VCP. mpremote can
  fail with "could not enter raw repl" while a script floods stdout — use the
  protocol's `cam.reset()` instead, or hard-reset by replugging.
- If the stream channel goes dead (0 frames, lock NAKs), a device reset heals
  it; stale sessions from crashed host processes are the usual cause.

### Frame rate: `sensor.set_framerate()` is the throttle
The default capture mode idles at a **fixed ~117.6 fps** no matter what —
resolution (QQVGA→HD), pixel format, exposure time, and drawing overhead all
measure identical, because the ISP runs the sensor at a fixed rate and
`snapshot()` just dequeues frames. `sensor.set_framerate(n)` unlocks it:
requests are honored up to a ceiling of **~460 fps at QVGA** and **~235 fps at
VGA** (higher requests clamp). At 460 fps the max exposure is ~2 ms, so frames
get dark indoors. Benchmarks: `bench_fps.py`, `bench_framerate.py` (run with
`mpremote run <file>` while no protocol client is attached).

**`clock.fps()` is a cumulative average since the clock was created**, not an
instantaneous rate. Printing it makes any added load (e.g. a host attaching
and streaming) look like a slow continuous fps decay over hours — it isn't.
Measure windowed fps instead (`time.ticks_ms()` over the last N frames, as
`red_square.py` does). Verified: device holds a flat 460.8 fps at QVGA *while*
the host streams; stream-channel reads stay ~7 ms and the stream buffer only
ever holds the latest frame (no backlog).

The host protocol read path sustains ~55-60 fps unthrottled; `fb_webui.py`
deliberately caps reads at ~25-30 fps (each read locks the stream channel),
polls status at 2 Hz, refreshes the protocol session every 10 min (no device
reset, no video gap), and resets the device only if frames stall ~8 s.

### Second N6 (sensorhub camera) gotchas — details in sargineer
- **An inserted SD card silently disables `/flash/main.py` autorun**: boot cwd
  becomes `/sdcard` and `main.py` is looked up there (missing = silent no-op →
  bare REPL). Fix: 2-line chain-loader `/sdcard/main.py` that execs
  `/flash/main.py` (already on the 256 GB card).
- **`pyb.ADCAll`/`read_core_temp()` while the sensor streams HARD-HANGS the
  MCU**: USB disappears, power LED stays on, only a physical replug recovers,
  and the hang bypasses the OTA crash-rollback (it's a C-level fault, not an
  exception). Battery via `machine.ADC(machine.Pin.board.BAT_ADC)` is safe.
- **`secrets.py` is not OTA-managed** — after `dusty generate openmv_n6`, `mpremote cp` it to
  `/flash` and reset, or the app runs stale config (missing attrs surface as
  once-a-minute `loop error` prints, not crashes).
- MQTT telemetry publishes to `home/cam/n6cam/#` every 60 s; broker credential
  `n6cam` lives in sensorhub's `.env`/`passwd`/`acl`.

### Misc
- The device streams sensor frames even with **no script running** (idle ISP
  pipeline) — getting frames does *not* prove your script is alive.
- Framebuffer stream arrives JPEG-compressed (`format` 0x06060000); the
  library decodes to RGB888 via PIL.
- QVGA on this sensor is 320×200 (not 240). Device loop runs ~120 fps; host
  read path sustains ~55-60 fps.
- `mpremote run --no-follow script.py` launches a script detached, but a later
  protocol connect/reset will kill it — flash `main.py` is the reliable way to
  keep a script running.
