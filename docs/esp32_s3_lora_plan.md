# DustyCam v1 — ESP32-S3 Camera Node with LoRa Reporting

Battery-friendly AI camera node: an ESP32-S3 camera board runs on-device
object detection and reports *detections* (not images) over LoRa/Meshtastic
to a base receiver. This is the microcontroller-class DustyCam target — no
Pi, no WiFi dependency, deployable anywhere in radio range.

## Architecture

```
┌─ camera node ──────────────────────────┐        ┌─ base ─────────────┐
│  ESP32-S3 cam board      Heltec V4     │  LoRa  │  Heltec V4 #2      │
│  capture → detect ──UART──► Meshtastic ├─ ~~~ ──►  Meshtastic app /  │
│  (FOMO / ESP-DL)         serial module │  915MHz│  serial logger     │
└────────────────────────────────────────┘        └────────────────────┘
```

Detections are a few bytes (`person 0.87 12,40`) — a good match for LoRa's
~1–5 kbps effective throughput. Images stay on the node (optional SD later).

Why two chips: the Heltec V4's ESP32-S3FN8 has **no PSRAM**, so it can't
host the camera framebuffer or a useful model. The split also keeps the
radio side on stock Meshtastic — zero radio firmware to write in v1.

## Phase 1 — breadboard prototype (all parts owned)

Hardware (measured facts for all three were kept in the parts warehouse):

| Role | Part | Notes |
|---|---|---|
| Camera + detection | XIAO ESP32S3 Sense (preferred) or Waveshare ESP32-S3-CAM-GC0308 | XIAO: best Edge Impulse FOMO support, 8MB PSRAM; needs headers soldered. Waveshare: already characterized in `~/code/wavesharecam_sandbox` (LESSONS.md — mpremote wedges it; currently MicroPython, needs Arduino/IDF reflash for detection) |
| Radio | Heltec WiFi LoRa 32 V4 (unit w/ pre-soldered headers) | stock Meshtastic, serial module enabled |
| Base | Heltec WiFi LoRa 32 V4 #2 | stock Meshtastic + phone app |

Wiring (3.3V logic both sides, no level shifting):

```
cam TX ──► Heltec RX   (free header GPIOs, e.g. 45/46 — set in Meshtastic)
cam RX ◄── Heltec TX
   GND ─── GND         (required even with separate USB power)
```

Steps:

1. **Detection firmware on cam board.** Edge Impulse FOMO (96×96 or
   160×160 grayscale), Arduino framework. Train on a few hundred images of
   the target scene/objects — candidate to reuse the dustycam data-gen +
   finetune pipeline with an Edge Impulse/TFLite-micro export target.
   Output: one text line per detection over UART @115200:
   `<class> <confidence> <cx>,<cy>`.
2. **Heltec node config.** Meshtastic serial module, `TEXTMSG` mode,
   RX/TX = chosen pins. Each line from the cam becomes a mesh message.
3. **Base station.** Second V4 paired to phone app; verify detections
   arrive end-to-end. Measure latency and range.
4. **Duty cycle sanity.** Rate-limit transmissions in cam firmware
   (e.g. min 10 s between reports, or report-on-change) — LoRa airtime
   is the scarce resource.

Exit criteria: object enters scene → mesh message on phone within ~5 s,
repeatable outdoors at target range.

## Phase 2 — custom board

Collapse to one chip: **RAK3112 WisDuo module** (~$8) = ESP32-S3 with
16MB flash + 8MB PSRAM + SX1262 in one solderable module, US915-capable,
Meshtastic-targeted.

Bring-up hardware before layout (~$30):

- RAK3112 **Breakout Board, 900 MHz variant** — $16 from RAK store
  (includes LoRa + WiFi PCB antennas w/ MHF4, headers, USB cable)
- Waveshare **OV2640 Camera Board** (~$10–13, LCSC/Amazon) — DVP camera on
  2.54 mm header with onboard regulators; jumper-wires to the breakout.
  Avoid bare 24-pin FPC "goldfinger" modules at this stage (need ZIF).
- Keep camera jumpers ≤10 cm; if frames are garbage, drop XCLK to 10 MHz.

Custom PCB scope: RAK3112 + 24-pin FPC camera connector + 3.3 V reg +
1S LiPo charger + MHF4 antenna. Camera pin map is free-choice on the S3
(LCD_CAM via GPIO matrix) minus the module's internal SX1262 pins (fixed,
per RAK datasheet).

Firmware decision to make in phase 2: keep detections + radio in one
custom firmware (esp32-camera + ESP-DL/EI + RadioLib), or fork Meshtastic
and add the camera task. Prototype learnings (model, pin map, protocol)
carry over either way — same S3 core.

## Risks / open questions

- FOMO grayscale accuracy on the actual target objects — validate early
  with a quick collected dataset before investing in the pipeline.
- Heltec V4 free-GPIO choice for serial module — confirm chosen pins are
  unused by OLED/SX1262/USB on V4 (V3 pinout is the reference; V4 is
  pin-compatible).
- Power budget: v1 prototype is USB-powered; deep-sleep + PIR wake is a
  v2 question.
- RAK3112 is new (2025) — check community Meshtastic/board-support
  maturity before committing the custom board to it.

## References

- RAK3112 breakout: https://store.rakwireless.com/products/rak3112-breakout-board-esp32-s3-sx1262
- Waveshare OV2640 board: https://www.waveshare.com/ov2640-camera-board.htm
- Meshtastic serial module: https://meshtastic.org/docs/configuration/module/serial/
- Seeed XIAO ESP32S3 Sense wiki: https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/
- Owned-part facts: measured `part.yaml` for amz-heltec-lora-v4, amz-xiao-esp32s3-sense,
  amz-esp32s3-cam-gc0308 (kept in the parts warehouse, not on this machine)
