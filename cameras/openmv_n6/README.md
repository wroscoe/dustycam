# openmv_n6

An **OpenMV N6** camera running MicroPython, uploading to the local sensorhub
with motion gating, MQTT telemetry, and SD offline buffering. Updates ship
over WiFi via a small OTA bootstrap rather than a USB reflash.

Moved in from `~/code/openmvsandbox`. Protocol notes are in
[`../../docs/openmv_n6_protocol.md`](../../docs/openmv_n6_protocol.md), and
`CLAUDE.md` here holds the working notes for this board.

## Layout

| Path | Contents |
|---|---|
| `software/sensorhub_cam.py` | The main app, deployed as `/flash/app.py`. `sensorhub_cam_lp.py` is the low-power variant. |
| `software/ota_main.py`, `ota.py`, `ota_push.py` | OTA bootstrap (`/flash/main.py`) and the host-side pusher. |
| `software/fb_webui.py`, `stream_server.py`, `camera_stream.py` | Framebuffer web UI and streaming helpers. |
| `software/bench_*.py`, `diag_backlog.py` | FPS/framerate benchmarks and backlog diagnostics. |
| `software/red_square*.py`, `main_factory_backup.py` | Minimal test scripts and the stock firmware's `main.py`. |
| `software/omv_patches.py` | Local patches to OpenMV behavior. |
| `tests/` | Empty — benchmarks live in `software/` and need a board. |
| `hardware/` | Empty — no board-specific design work yet. |

## Deploying

```bash
cd software
./ota_push.py            # push a new app.py to the N6 over WiFi
```

## Credentials

Credentials come from `~/.dusty/` — see the [repo README](../../README.md).
`software/secrets.py` is **generated** from it and gitignored:

```bash
dusty generate openmv_n6
```

It carries the WiFi credentials, sensorhub host/port, MQTT login and the OTA
token, and is deployed to `/flash/secrets.py` on the board — the same
convention the ESP32-S3 camera uses. Note `secrets.py` is *not* OTA-managed:
after regenerating, `mpremote cp` it to `/flash` and reset, or the app runs
stale config. Edit `~/.dusty/secrets.toml` and regenerate — never edit the
generated file.
