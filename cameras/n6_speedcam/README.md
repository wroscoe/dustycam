# n6_speedcam

An **OpenMV N6** camera paired with a **Hi-Link HLK-LD2415H** 24 GHz
vehicle-speed radar: the radar reports every passing vehicle's speed and
direction over serial and wakes the camera, the camera captures the vehicle
and reports both to sensorhub. Solar powered, in one weather-resistant
printed box on a tripod stud.

This is a new camera directory rather than a variant of
[`../openmv_n6/`](../openmv_n6/) because the radar changes the hardware
(carrier board, power, enclosure) and the software's trigger model. The N6
firmware pieces that are board-generic — the OTA bootstrap, the sensorhub
uploader, the framebuffer tools — stay in `openmv_n6/software/` and are
reused, not copied.

## Layout

| Path | Contents |
|---|---|
| [`hardware/`](hardware/) | Radar facts + datasheet, the tscircuit carrier board, the solar/LiPo power design and budget, the printed enclosure. Start at `hardware/README.md`. |
| [`software/ld2415h.py`](software/ld2415h.py) | MicroPython driver: ASCII speed-frame parser (host-testable) + `CF` command builders + a `Radar` class on `machine.UART`. |
| [`software/speedcam.py`](software/speedcam.py) | App skeleton for `/flash/app.py`: configures the radar, captures at the peak speed of each pass, optional deep sleep with wake on `P11`. Upload/MQTT/SD buffering to be merged in from `openmv_n6/software/sensorhub_cam.py`. |
| [`tests/`](tests/) | Host tests for the parser and command builders (`python -m pytest cameras/n6_speedcam/tests`). |

## Wiring in one table

| Signal | Radar J4 | Carrier | N6 |
|---|---|---|---|
| speed frames | pin 4 `UART_TX` | JR.TX → JN.RX | `P5` (UART3 RX) |
| config commands | pin 3 `UART_RX` | JR.RX ← JN.TX | `P4` (UART3 TX) |
| detection → wake | pin 1 `PULSE` *or* pin 2 `AA` | Q1 inverter / direct → `WAKE` | `P11` (WKUP3, active low) |
| 12 V | pin 8 `VCC`, pin 7 `GND` | JP ← DFR0535 OUT3 | — |
| 5 V | — | JP ← DFR0535 OUT1 → JN.VIN | `VIN` (4.7–5.7 V) |
| 3.3 V ref | — | JN.3V3 → WAKE pull-up | `3.3V` |

The radar's PULSE pin swings to its **supply voltage (9–24 V)** — it goes
through the carrier's transistor, never straight to the N6.

## Status

Designed 2026-09-02, nothing built yet. Order of operations when the parts
arrive:

1. Bench the radar alone on the DFR0535's 12 V with a USB-serial adapter
   (3.3 V!) and confirm the frame format and which trigger pin the module
   populates. Record what you learn in sarg — it has nothing on this module.
2. Build the carrier on perfboard from `hardware/carrier/README.md`.
3. Run `software/speedcam.py` with `SLEEP = False` over USB; then try
   `SLEEP = True` and see whether `P11` really wakes OpenMV 5.0 firmware.
4. Measure the draws listed in `hardware/power/README.md` and re-run
   `budget.py` before buying the panel and pack.
5. Caliper the radar's connector and the DFR0535's holes, update the two
   `ref/` models, re-run `verify_n6_speedcam.py`, print.

## Standard mapping (docs/camera_standard.md, 2026-09-02)

To be built by following `docs/camera_recipe.md` end to end (phase 5): it is
the recipe's first test. Today: `software/speedcam.py` is a skeleton
(radar → peak speed → `img.save` to SD), `software/ld2415h.py` is the tested
radar parser, `SLEEP`/deep-sleep wake on `P11` is untested on OpenMV 5.0.

| Stage / feature | Planned |
|---|---|
| Sense | radar pass: `radar_kmh`, `radar_dir` in meta and telemetry; battery/charge from the DFR0535 |
| Watch | `why: sensor` from the radar; heartbeat |
| Capture | full-res at the radar's quiet edge (500 ms after the last frame of a pass) |
| Judge | none at first (server analyzer); vehicle gate later |
| Deliver / Report / Serve | shared MicroPython modules from phase 2 (gate, spool, telemetry, config pull, firmware pull, control port) |
| Rest | deep sleep, radar wake; `idle_s` |
| Setup mode | page with preview + focus score + live radar readout so the radar and lens can be aimed together |
