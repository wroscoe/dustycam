# openmv_n6

An **OpenMV N6** camera running MicroPython, uploading to the local sensorhub
with motion gating, MQTT telemetry, and SD offline buffering. Updates ship
over WiFi via a small OTA bootstrap rather than a USB reflash.

Moved in from `~/code/openmvsandbox`. Protocol notes are in
[`../../docs/openmv_ide_protocol_v2.md`](../../docs/openmv_ide_protocol_v2.md) (the OpenMV IDE wire protocol, used by the USB preview tools), and
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

Copy `software/secrets_example.py` to `software/secrets.py` and fill it in;
it is gitignored. Keep the master values in `~/.dusty/` — see the
[repo README](../../README.md).

It carries the WiFi credentials, sensorhub host/port, MQTT login and the OTA
token, and is deployed to `/flash/secrets.py` on the board — the same
convention the ESP32-S3 camera uses. Note `secrets.py` is *not* OTA-managed:
after editing, `mpremote cp` it to `/flash` and reset, or the app runs stale
config.

## Standard mapping (docs/camera_standard.md, 2026-09-02)

Status: built, silent since 2026-08-20. Phase 3 of the plan brings it onto
the shared MicroPython modules; needs the board on the desk.

| Stage / feature | Today | Gap |
|---|---|---|
| Boot | hand-copied secrets.py; `ota_main.py` recovery | dustygen; stored server config |
| Connect | WiFi, NTP, `ip` in meta | — |
| Sense | `BAT_ADC` (divider 1.5 is a guess), `CHG`, IMU speed estimate | measure the divider |
| Watch | VGA diff + heartbeat | `why` |
| Capture | the same VGA frame that was diffed | full-res capture (as rt1062 1.5-rt) |
| Judge | none | not applicable on-device |
| Record | meta `ts,w,h,diff,heartbeat,buffered,v,ip` | `seq,cfg,mode,why,gate` |
| Deliver | ingest :8088 direct, no token, no TLS; SD spool | blob gate |
| Report | direct MQTT (ACL-limited, silent drops) | HTTP `/telemetry` via gate |
| Serve | push OTA :8266 | pull config/firmware, status/setup port |
| Rest | idle loop; LP variant turns WiFi off between deliveries | LP becomes a power setting, not a second file |
| Setup mode | none (USB preview via fb_webui.py only) | the whole thing |
| Layout | `software/` flat | `software/app|host`; `tests/` is empty |
| Deny list | `pyb.ADCAll`/`read_core_temp` hangs the MCU while streaming; `cpufreq` gated | record in board facts |
