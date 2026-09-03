# HLK-LD2415H — 24 GHz vehicle speed radar

Everything the rest of this tree relies on, pulled from the Hi-Link datasheet
v2.0 (2023-10-22), kept here as
[`hlk-ld2415h_datasheet_v2.0.pdf`](hlk-ld2415h_datasheet_v2.0.pdf) (image-only
PDF; the numbers below were read off it by eye, so check anything load-bearing
against the scan). Sarg has nothing on this module or any HLK radar
(searched 2026-09-02: `LD2415H`, `HLK radar UART`, 0 hits) — worth recording
the first bring-up lesson there.

## Electrical

| Item | Value |
|---|---|
| Supply | **9–24 V DC**, 12 V typical, ≤ 50 mA @ 12 V (≤ 1.2 W actual; 20 dBm EIRP) |
| Serial | TTL **3.3 V** UART, **9600 8N1, fixed**; RS-485 A/B in parallel on the same pins |
| Speed range | 1–240 km/h, ±1 km/h (at 0° between beam axis and travel) |
| Range | ≥ 180 m on cars ("1 km" is marketing) |
| Beam (3 dB) | horizontal **40°**, vertical **16°** — the long (69 mm) axis of the board must be **vertical** for the wide axis to be horizontal |
| Frequency | 24.075–24.175 GHz, 24.125 typ. (CW Doppler) |
| Temperature | −20…+85 °C operating |

### J4 connector (8-pin, 1.25 mm pitch; test kit ships a Dupont pigtail)

Pin order is the datasheet's table, viewed from the **back** of the module.

| Pin | Name | Function | Carrier net |
|---|---|---|---|
| 1 | PULSE | Detection output, **active HIGH at VCC level** (PNP AO3407 from VCC — 9–24 V, *not* 3.3 V safe). "High-level version" boards only. | `PULSE` → 47 k → Q1 base (inverted onto `WAKE`) |
| 2 | AA | Detection output, **active LOW, open collector** (MMBT3904). "Low-level version" boards only. | `WAKE` (10 k pull-up to N6 3V3) |
| 3 | UART_RX | radar receives (3.3 V) | N6 **P4 / UART3 TX** |
| 4 | UART_TX | radar transmits (3.3 V) | N6 **P5 / UART3 RX** |
| 5 | B | RS-485 B | n/c |
| 6 | A | RS-485 A | n/c |
| 7 | GND | | GND |
| 8 | VCC | 9–24 V | 12 V (DFR0535 OUT3) |

Only one of PULSE / AA is populated on a given module (datasheet §5: "the
radar board is the high-level active output signal version" vs "active low
output signal version"). The carrier conditions both so it does not matter
which one you received; the unpopulated pin floats harmlessly. The trigger
fires when the measured speed exceeds the threshold set with command 0x04
(below), which is what lets the N6 sleep between vehicles.

## Serial protocol

### Speed frames (radar → host), 9 bytes, continuous while a target is tracked

```
V + 0 1 2 . 3 \r \n      approaching target, 12.3 (units per command 0x02)
V - 0 0 1 . 9 \r \n      receding target
```

ASCII: `V`, sign (`+` coming / `-` going), hundreds, tens, units, `.`,
tenths, `0x0D 0x0A`. Nothing is sent when no target is in view — the host
must time out to zero. Default rate ≈ 11 frames/s (setting 0x01); 0x00 ≈ 22/s.

### Commands (host → radar)

All commands: `43 46 <fn> <p1> <p2> <p3> 0D 0A` (`"CF"` + function code +
three parameters + CRLF). Default settings shown.

| fn | Purpose | p1 | p2 | p3 |
|---|---|---|---|---|
| 0x01 | detection tuning | min speed km/h (**0x01**) | angle compensation, degrees (**0x00**) | sensitivity 0x01–0x0F (**0x05**; *smaller* = more sensitive/longer range, larger = more interference-tolerant) |
| 0x02 | output mode | direction: **0x00** both, 0x01 approaching only, 0x02 receding only | rate: 0x00 ≈ 22 fps, **0x01** ≈ 11 fps, each +1 halves again | unit: **0x00** km/h, 0x01 mph, 0x02 m/s |
| 0x03 | anti-vibration coefficient 0x00–0x70 (**0x00**) | — | — |
| 0x04 | trigger output (PULSE/AA) | hold time, s, 0x00–0xFF (**0x00** = off) | speed threshold km/h (**0x00**) | — |
| 0x05 0x01 | switch to custom protocol mode (11-byte form: `43 46 05 01 00 00 00 00 00 00 00`) | | | |
| 0x07 | read settings (13-byte form `43 46 07` + ten `00`, standard mode only) → `No.: 20210726 v3.0 X1:.. X2:.. …` | | | |

Datasheet examples: `43 46 01 03 0A 05 0D 0A` = min 3 km/h, 10° angle
correction, sensitivity 5; `43 46 02 01 02 00 0D 0A` = approaching only,
≈5.5 fps, km/h. The settings persist across power cycles (they are what the
vendor PC tool edits over RS-485). The MicroPython driver in
[`../../software/ld2415h.py`](../../software/ld2415h.py) wraps all of this.

Standard ↔ custom protocol mode switch: `FA 31 30 30 FB` (wait for the
reply) then `FA 55 AA FF FB`. Leave it in standard mode.

## Mechanical

| Item | Value |
|---|---|
| Board | **69 × 53 mm**, 5 mm thick including back-side parts; antenna patches on the front face, 4 rounded corners |
| Mounting | 4 × Ø2.75 holes on a **48.5 × 64.5 mm** pattern (2.25 mm in from each edge); M2.5 |
| Connector | J4 on the back face, **measured 2026-09-02**: 14.8 mm wide, near side 7.3 mm in from the short edge it parallels, one end 8.2 mm from the (back-view) left edge. Depth and height still assumed 6 × 6 mm |
| Install | 1–2 m high beside the road, antenna face < 10° off the travel direction, long axis vertical |

The antenna face wants a plain dielectric in front of it and nothing metal
within a few cm. See the case README for the radome thickness rule.

## Ordering

Amazon/AliExpress listing name: "HLK-LD2415H 24G Millimetre Wave Vehicle
Speed Feedback Radar Module 1KM Long Range Speed Sensor Serial Communication".
The **test kit** (module + USB-RS485 dongle + Dupont pigtail) is the version
worth having; the bare module needs its own 1.25 mm pigtail. Hi-Link product
page: https://www.hlktech.net/index.php?id=1220.
