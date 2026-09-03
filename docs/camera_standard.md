# The camera standard

One loop, two modes, one contract, one layout. This document is the
reference; [`camera_recipe.md`](camera_recipe.md) is the brief for writing a
new camera against it, and
[`camera_standard_proposal.md`](camera_standard_proposal.md) records the
survey and the decisions behind it (2026-09-02).

It is written for cameras but assumes nothing camera-specific: a device
without an imager (a sensor node) implements the same stages and marks
Watch / Capture / Judge as not applicable.

## 1. In one paragraph

A camera runs **one loop** in one of two modes. In **live** mode it watches
cheaply, captures when triggered, judges, records a frame plus a standard
meta JSON, delivers it to sensorhub or spools it, reports telemetry on a
fixed cadence, and always keeps its control plane reachable. In **setup**
mode it stops recording and instead streams a live preview with a focus
score, takes manual shots, shows live sensor readings, and reflects new
configuration, then returns to live mode by itself. Configuration has three
tiers (identity and credentials; tuning; board facts) and only tuning
changes after flashing. Every camera speaks the same contract to sensorhub,
lives in the same folder layout, and states in its README which stages it
implements and which it does not.

## 2. The pipeline

Stages are named so a README, a log line and a test can refer to them. A
stage the board cannot do is stated as "not applicable: why" in the README,
never silently skipped.

**Boot.** Load identity and credentials from the secrets file. Load tuning:
the defaults stamped into the firmware, overridden by the last configuration
pulled from the server and kept in non-volatile storage. Check for a
recovery request: user button held, a flag file, or the previous app crashed
with nothing to roll back to. Recovery mode is a tiny program that only
serves the control plane and is never updated over the air.

**Connect.** Join WiFi (or whatever radio the board has). Sync the clock.
Note the board's IP; it goes into every meta and every telemetry report.
Failure to connect is not fatal: the loop runs and spools.

**Announce.** Send one telemetry report: firmware version, config version,
mode, uptime zero, boot reason.

Then loop:

1. **Sense.** Read the sensors the device has (radar, battery, charge
   state, IMU, temperature). Readings are numbers with stable names.
2. **Watch.** Take a cheap preview frame (small, grayscale or RGB565) and
   evaluate the trigger: motion (difference against the last *recorded*
   frame), a sensor event (radar pass, PIR), an interval, a manual request
   from setup mode, or the heartbeat timer. Every trigger has a name
   (`why`, section 5).
3. **Capture.** Take the real frame at the configured quality (full sensor
   resolution, sensor-side JPEG when the sensor can). Note width, height and
   byte size.
4. **Judge.** Optionally run an on-device model on the frame or the preview.
   The result is a list of detections (label, confidence) and a decision:
   keep, discard, or keep-as-audit (one in N discarded frames still sent so
   the gate can be checked from the server).
5. **Record.** Build the meta JSON with the standard keys plus the device's
   own sensor fields. Build the telemetry dict from counters and readings.
6. **Deliver.** Upload frame and meta to sensorhub. If the server is
   unreachable, spool to local storage: sidecar JSON first, then the image
   to a temp name, then rename. When an upload succeeds again, pause
   recording and drain the spool oldest-first, then resume. Delivery is
   at-least-once; the server de-duplicates on `ts`.
7. **Report.** On a fixed cadence (`telemetry_s`) send the telemetry dict.
   Numbers only: counters (sent, skipped, failed, pending), health (memory,
   RSSI, uptime, battery), the current mode and config version, sensor
   readings.
8. **Serve.** Service the control plane: status, mode change, config
   refresh, firmware check. On an always-on board this is a non-blocking
   poll from *every* place the loop spends time, including mid-drain. On a
   wake-cycle board it happens once per contact, right after a successful
   upload. Nothing may block the control plane for more than a few seconds.
9. **Rest.** Always-on boards idle until the next period, keeping the
   preview stream fresh. Wake-cycle boards persist state (last thumbnail,
   counters, config) and deep-sleep until the next wake or sensor interrupt.

### The laws

- **The heartbeat frame is the safety net.** It is the only guarantee that a
  board with a bad config or a broken gate stays reachable.
- **Firmware first, then config.** Never publish a config value the running
  firmware does not understand.
- **`ts` is the truth; `ip`, `v`, `why` are always present** in meta.
- **A failed publish must be visible.** Telemetry goes over HTTP to the gate
  (which answers), never directly to MQTT (which drops silently on an ACL miss).
- **No TLS while a viewer is attached** to the preview; a publish blocks the
  loop for seconds.
- **The spool never fills storage:** cap on pending files, oldest reclaimed.
- **Board-health calls known to hang the MCU** are on the board's deny list
  in board facts and are never called from the loop.
- **Recovery mode is never updated over the air.**
- **The control plane is polled from everywhere the loop waits.**

## 3. Modes

| Mode | What runs | Entered by | Left by |
|---|---|---|---|
| **live** | the loop | boot (default); leaving setup | entering setup; crash → recovery |
| **setup** | preview with focus score; manual shots; sensor readouts; config refresh; control plane | user button held ≥1 s; `GET /setup` on the board; server config `mode: setup` at the next contact | timeout (`setup_secs`, default 300); `GET /live`; button again; server config back to `live` |
| **recovery** | control plane only | crash with nothing to roll back to; button held at boot | a successful firmware update |

**The setup page** (served by the board, viewed on a phone on the LAN):

- Live preview, grayscale, small, 5-10 fps, with the sharpness score of a
  centre region (stdev of a ×4 Laplacian), the session's best, seconds
  remaining. The page polls `/status` every 2 s and reopens the stream if
  frames stop; phones drop long connections.
- **Shoot**: capture and deliver one full-quality frame now (`why: manual`).
- Live sensor readouts from the status poll (radar speed, battery, RSSI).
- Current tuning and `cfg`, and **refresh**: pull `/config/<device>` now.
- Links to the device's sensorhub page and firmware version.

The setup page **edits nothing on the board**. Tuning is edited on the
workstation and pulled (section 6). Decision 2026-09-02.

While in setup: recording paused, heartbeat timer held, telemetry continues
with `mode=1` only while no viewer is attached, OTA reachable, LED shows the
mode. On exit: motion reference reset, one confirmation frame delivered.

Wake-cycle boards: setup means "stay awake for `setup_secs` after this
contact and serve the page"; deep sleep resumes when it ends. Raspberry Pi:
the same page from the same service; the preview is the pipeline's preview
branch.

## 4. The contract (device ↔ sensorhub)

**Identity.** `device` is a short lowercase id (`rt1062cam`, `n6cam`,
`goouuu1`, `speedcam1`). It names the blob path, the telemetry topic prefix,
the config file, the firmware artifact, and the `devices.json` entry.

**Frames.** `POST /blob/<device>/frame`, `Content-Type: image/jpeg`, body =
image, header `X-Meta` = JSON. Through the blob gate (`:8089` LAN, Funnel
`:10000`, `X-Token`) by default; ingest `:8088` direct is a LAN-only
fallback. Raw-socket HTTP on MicroPython (the frozen `requests` cannot read
the ingest's HTTP/1.0 reply).

**Meta keys** (every board):

| key | meaning |
|---|---|
| `ts` | capture time, epoch seconds (authoritative) |
| `seq` | monotonic per boot |
| `w`, `h` | pixels |
| `v` | firmware/app version string |
| `cfg` | config version the board is running |
| `ip` | board IP |
| `mode` | `live` / `setup` / `recovery` |
| `why` | `motion` / `heartbeat` / `sensor` / `interval` / `manual` / `audit` / `burst` / `boot` |
| `diff` | motion fraction that fired (0 if not motion) |
| `gate` | threshold in force |
| `heartbeat`, `buffered` | booleans (existing analyzer and pages use them) |
| `det` | optional on-device detections `[{"label":…, "conf":…}]` |
| device-specific | prefixed by sensor name: `radar_kmh`, `radar_dir`, `batt_v`, `lum`, `exp` |

Reserved for the server (the analyzer merges them in): `detected`,
`animals`, `detections`. Boards never write them.

**Telemetry.** `POST /telemetry/<device>`, JSON object of numbers; the gate
republishes each key as MQTT `<device>/<key>` `{"v": n}`. Standard keys:
`uptime_s`, `mem_free`, `rssi`, `frames_sent`, `frames_skipped`,
`upload_failures`, `pending_files`, `mode` (0 live, 1 setup, 2 recovery),
`cfg`, `boot_count`; plus `batt_v`, `charging`, and sensor readings by name.

**Config pull.** `GET /config/<device>` → JSON of tuning keys plus `cfg`.
The board applies known keys, ignores unknown ones, stores the result, and
reports `cfg` in meta and telemetry. Pulled at boot, after every successful
upload on wake-cycle boards, every `config_s` (default = `heartbeat_s`) on
always-on boards, and on **refresh** from the setup page.

**Firmware.** Pull is the standard (decision 2026-09-02): the board checks
`GET /firmware/<device>/version` on every config pull, compares with its
own, fetches `GET /firmware/<device>.py|.bin`, verifies (compile check for
MicroPython, image validation for ESP-IDF), installs with rollback (previous
copy kept; the new image must complete one successful upload before it is
marked good). Staging on the workstation = copy the artifact and a
`.version` file to `/hd2/sensorhub/firmware/`. The OpenMV push path
(`ota_push.py` → `POST /update` on the board) stays as a developer fast
path on the LAN, not as the standard.

**Control plane on the board** (always-on boards; decision 2026-09-02): one
HTTP listener on `:8266`. Open on the LAN: `GET /status`, `/setup`, `/live`,
`/stream`, `POST /shoot`, `POST /refresh`. Token (`OTA_TOKEN`) required only
for `POST /update`. The gate token protects the public path; the board's
listener is LAN-only.

**Server side.** The gate proxies `GET /config/<device>` and
`GET /firmware/<device>[/version|.py|.bin]` (token-checked) so remote boards
can pull. `devices.json` carries `expect` for every device. The device page
shows `mode` and links to the board's setup page when it is on the LAN.

## 5. Configuration: three tiers

| Tier | Examples | Lives in | Reaches the board by | Changes after flashing? |
|---|---|---|---|---|
| **identity + credentials** | device id, WiFi, server host/ports, tokens, MQTT creds | `~/.dusty/secrets.toml` + `[server]` in `config.toml` | generated secrets file, flashed over USB (or compiled in) | no |
| **tuning** | period, diff threshold, heartbeat, telemetry cadence, capture size, setup secs, gate pct, wake interval, mode | `~/.dusty/config.toml` `[camera.<name>]` | stamped into the firmware as defaults **and** published to `/data/config/<device>.json` for pull | yes, without reflash |
| **board facts** | pins, sensor, framesize names, LED/button names, deny-listed calls | the board adapter in the camera's software | with the firmware | with the firmware |

`tools/dustygen <camera>` (replaces the removed `dusty generate`) reads
`~/.dusty` and the camera's `camera.toml`, writes the secrets file in the
board's format (`secrets.py`, `sdkconfig.secrets`, `.env`), stamps the tuning
defaults into the app, and writes the server config file. `--public`
selects the Funnel host/port and TLS. A tuning change is: edit
`config.toml`, run `dustygen`, and the board pulls it at its next contact.

## 6. Folder layout

```
cameras/<camera>/
  README.md            what it is, board facts, deploy, a status line, and a
                       "Standard mapping" section: one line per stage saying
                       how this camera does it or "not applicable: why"
  camera.toml          manifest (below)
  hardware/            case, carrier, power, optics: sources + exports, README
  software/
    app/               what runs on the board: one app; the power model is a
                       setting, not a second file
    host/              workstation tools: deploy, stage-firmware, bench, preview
  tests/               host-runnable pytest: parsers, trigger math, meta
                       builder, config merge, spool naming. No board needed.
```

`camera.toml`:

```toml
id = "rt1062cam"
board = "OpenMV Cam RT1062 R6"
runtime = "micropython"        # micropython | espidf | cpython
power = "always_on"            # always_on | wake_cycle
status = "live"                # designed | built | live | archived
sensors = ["battery", "charge"]
capabilities = ["preview", "focus_score", "motion", "full_res", "spool",
                "pull_config", "pull_firmware", "push_firmware", "setup_mode"]

[tuning]                       # defaults; the same keys are allowed in config.toml
period_s = 10
diff_min_frac = 0.04
heartbeat_s = 300
telemetry_s = 60
setup_secs = 300
```

Shared code: `cameras/common/<runtime>/`.

- `common/micropython/`: `uplink.py`, `spool.py`, `telemetry.py`,
  `control.py` (status/setup/stream listener), `motion.py`, `focus.py`,
  `otapull.py`, `config.py`, and `bundle.py`, which inlines the modules a
  camera lists into a single `app.py` at build time because the OTA channel
  installs one file. Tests run on the modules.
- `common/espidf/components/`: `dusty_uplink`, `dusty_spool`,
  `dusty_config`, `dusty_ota`; a camera's `main.c` stays thin.
- `common/cpython/dusty/`: the same stages as a package for the Pi.

## 7. Reference implementation

`cameras/openmv_rt1062` (2026-09-03): `software/app/board.py` + `app.py` on
`cameras/common/micropython/` (uplink, spool, config, otapull, motion,
camera, focus, control), bundled by `bundle.py`, generated and staged by
`tools/dustygen`. Two design notes that other MicroPython cameras inherit:

- **Port takeover.** The loader's `ota.py` listener on `:8266` is closed by
  `control_init()` and the app binds the same port, so the loader (never
  updated OTA) needed no change and `ota_push.py` still works.
- **Prove-out without touching the loader.** `otapull` writes
  `fw_pending.txt` on install; at boot `fw_boot_check()` sees a pending
  version that is not itself, concludes the loader rolled it back, and
  blacklists it in `fw_bad.txt`; the first successful upload clears pending.

## 8. Definition of done for a camera

- [ ] `camera.toml` present and true; README status line and "Standard mapping" table filled in.
- [ ] Frames arrive with every standard meta key; telemetry with every standard key.
- [ ] Config pull works and `cfg` changes without a reflash.
- [ ] Firmware pull with rollback proven with a deliberately bad build.
- [ ] Spool and drain proven by unplugging the network for ten minutes.
- [ ] Setup mode from button and URL; page reconnects after the phone locks; timeout returns to live.
- [ ] Heartbeat every `heartbeat_s` for a full day.
- [ ] `devices.json` entry with `expect`.
- [ ] Host tests pass with no board attached.
- [ ] Lessons recorded in sarg.
