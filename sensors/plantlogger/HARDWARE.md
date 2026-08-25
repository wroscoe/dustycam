# Plant Logger — Hardware Documentation

Last verified: 2026-07-21

## System overview

```
┌─────────────────────────────┐        Wi-Fi (2.4 GHz)        ┌──────────────────────────┐
│  FeatherS3[D]               │  "avenueofgiants" (main SSID) │  Workstation "homegpu"   │
│  ├─ soil sensor (STEMMA)    │ ────────────────────────────► │  192.168.86.26           │
│  ├─ ambient light (onboard) │   POST /api/reading           │  plantlog server :8087   │
│  ├─ MAX17048 fuel gauge     │   hourly + offline queue      │  SQLite + dashboard      │
│  └─ 1200 mAh LiPo           │                               │  (systemd user service)  │
└─────────────────────────────┘                               └──────────────────────────┘
```

## Microcontroller — Unexpected Maker FeatherS3[D]

The **[D]** (dual-antenna, 2025 revision) — NOT the original FeatherS3. Several
pin functions differ; using original-FeatherS3 docs/helpers for battery sensing
was the source of a long debugging session (see Quirks).

| Item | Value |
|---|---|
| SoC | ESP32-S3 (QFN56, rev v0.2), 2.4 GHz Wi-Fi + BLE |
| USB id (MicroPython app) | `303a:80d7` "Unexpected Maker FeatherS3" |
| USB id (ROM bootloader) | `303a:1001` "Espressif USB JTAG/serial debug unit" |
| MAC / device id | `44:1b:f6:dc:a2:44` → dashboard device id `441bf6dca244` |
| Serial port | `/dev/ttyACM0` (stable: `/dev/serial/by-id/usb-Unexpected_Maker_FeatherS3_441bf6dca2440000-if00`) |
| Firmware | MicroPython v1.28.0, official `UM_FEATHERS3` build (no dedicated S3D build exists; fine except battery helper — see Quirks) |
| Charging | USB-C, onboard 1S LiPo charger; amber CHG LED next to USB |
| Deep sleep | ~tens of µA; used between readings on battery power |

### Pin map (as used by this project)

| Pin | Function |
|---|---|
| IO8 / IO9 | I2C1 SDA / SCL — **onboard MAX17048 fuel gauge @ 0x36** + always-powered STEMMA QT port |
| IO16 / IO15 | I2C2 SDA / SCL — second STEMMA QT port (LDO2-powered) — **soil sensor lives here** |
| IO2 | MAX17048 interrupt line (**not** VBAT sense — that's the original FeatherS3) |
| IO4 | Ambient light sensor (onboard, ADC1_CH3, raw counts) |
| IO13 | Blue LED (flashes briefly during each logging cycle) |
| IO39 | LDO2 enable — powers the I2C2 STEMMA port and NeoPixel; off in deep sleep |
| IO40 | RGB NeoPixel data (unused by logger) |
| IO41 | RF switch: HIGH = external u.FL antenna, LOW = onboard antenna (software-selectable, no soldering) |

## Sensors

### Adafruit STEMMA Soil Sensor (capacitive, seesaw-based)

- I2C address **0x36** on **I2C2** (SDA=IO16, SCL=IO15), plugged into the
  LDO2-switched STEMMA QT port. Powered up ~300 ms before each reading, off after.
- Moisture: seesaw touch channel 0. Observed calibration: **~350 = in air/bone
  dry, ~1016 = firm grip/saturated**. Dashboard guides: <400 dry, >800 wet.
- Temperature: on-chip sensor, ±2 °C at best, reads a few degrees warm.

### Onboard ambient light sensor

Raw ADC counts from IO4 (0–4095-ish). Uncalibrated — useful for trends
(day/night, window vs shade), not lux.

### MAX17048 battery fuel gauge (onboard)

I2C **0x36 on I2C1** (IO8/IO9) — same address as the soil sensor, which is why
the two sensors MUST stay on separate buses. Registers used: VCELL (0x02,
78.125 µV/LSB), SOC (0x04, %/256), CRATE (0x16, 0.208 %/hr signed). SOC
recalibrates for ~an hour after each power-up; ignore early wobble.

## Power

- Battery: 1S LiPo pouch, **3.7 V nominal, 1200 mAh**, JST-PH, charges from USB-C.
- Measured full: 4.02–4.21 V. Runtime at hourly logging (deep sleep between):
  months. The board switches USB↔battery seamlessly without rebooting.
- **Do not** connect non-LiPo packs (e.g. AA packs) to the JST — the onboard
  charger will attempt to charge them.

## Host / server

- Workstation `homegpu`, LAN IP **192.168.86.26** (DHCP — consider reserving in
  Google Home app; the board's `secrets.py` hardcodes it).
- Plant logger server: Python stdlib, SQLite, port **8087**, systemd user
  service `plantlog.service` (enabled, linger on).
- Firewall: ufw rule `allow from 192.168.86.0/24 to any port 8087 proto tcp`
  ("plant logger") — required; ufw default-drops inbound.

## Network

- Router: Google/Nest WiFi. Main SSID `avenueofgiants` = 192.168.86.0/24,
  guest SSID = 192.168.87.0/24.
- **Guest network is isolated from main** — devices on guest cannot reach
  192.168.86.x. The sensor must join the MAIN network (it does), and phones
  must be on main Wi-Fi to open the dashboard.
- Board's observed RSSI at its current spot: -54 to -59 dBm (strong).
- ESP32-S3 radio is 2.4 GHz only — it cannot see or join 5 GHz.

## Quirks & lessons learned (read before debugging!)

1. **FeatherS3[D] ≠ FeatherS3 for battery sensing.** The original board has an
   analog divider on IO2; the [D] has the MAX17048 fuel gauge instead, and IO2
   is the gauge's interrupt line (idles at 3.3 V). The `feathers3` helper's
   `get_battery_voltage()` therefore returns a bogus ~4.86 V on this board
   (saturated ADC × scale factor). Always read the fuel gauge over I2C.
2. **0x36 address collision.** Fuel gauge (fixed 0x36, I2C1) and soil sensor
   (default 0x36) collide if the soil sensor is plugged into the I2C1 STEMMA
   port. Keep the soil sensor on the I2C2 port, or re-address it (solder its
   A0 jumper → 0x37).
3. **I2C2 pin order:** electrically SDA=IO16, SCL=IO15 — the pinout card lists
   IO15 first, which reads as if SDA=15. It isn't.
4. **Stuck in bootloader:** if the board enumerates as `303a:1001` and won't
   run firmware, it's in ROM download mode; esptool's software reset often
   can't exit it over USB-JTAG — press the physical RESET button once.
5. **mpremote sessions interrupt `main.py`** and leave the board idling at the
   REPL (LED frozen). Run `mpremote reset` after any manual session to resume
   logging.
6. **u.FL antenna** (future long-range work): on the [D] the antenna path is
   selected by IO41 in software — no 0-Ω resistor surgery like the original.

## On-board files (MicroPython flash)

| File | Purpose |
|---|---|
| `main.py` | boots into `logger.main()` after a 2 s grace period |
| `logger.py` | hourly cycle: sensors → `data.csv` → Wi-Fi POST; offline queue in `pending.jsonl`; errors in `errors.log` |
| `secrets.py` | Wi-Fi SSID/password + server URL (not in the repo copy? it is — treat the project dir as private) |
| `data.csv` | local log, rotates at 400 KB to `data.csv.old` |

Project source of truth: `~/code/feathers3-blink/` (board files in `board/`,
server + dashboard in `server/`, flashing tools in `.venv`).
