# n6_speedcam carrier v0.1 — OpenMV N6 ↔ HLK-LD2415H

The small board between the camera, the radar and the solar manager. It does
four things and nothing else: breaks the radar's 8-pin pigtail out onto a
labelled header, carries the 12 V and 5 V rails from the DFR0535 with bulk
capacitance at the radar end, turns the radar's detection output into an
active-low **wake** line for the N6's `P11 / WKUP3`, and routes the 3.3 V
UART straight through.

Designed **perfboard-first**, like `~/code/piezohat`: every part is
through-hole and every hole sits on one 2.54 mm lattice, so the PCB layout
below is a 1:1 placement map for a protoboard build. The gerbers in
`dist/n6_speedcam_carrier_gerbers.zip` still fab a real PCB.

**Status: unverified.** Designed 2026-09-02 from the LD2415H datasheet, the
N6 pinout diagram and the DFR0535 datasheet; netlist machine-checked with
tscircuit, never built or powered.

## Circuit

```
DFR0535 OUT3 (12 V) ──┬── C1 100µ ── C2 100n ──► radar VCC (pin 8)
DFR0535 OUT1 (5 V)  ──┬── C3 100µ ──► N6 VIN         ── R4 1k ── D1 (power LED)

radar AA    (pin 2, open-collector, active LOW) ─────────────┐
radar PULSE (pin 1, active HIGH at VCC) ── R2 47k ── Q1 B    ├── WAKE ──► N6 P11
                                          R3 100k B→GND, Q1 C ┘   R1 10k ► N6 3V3
radar TX (pin 4) ──► N6 P5 (UART3 RX)
radar RX (pin 3) ◄── N6 P4 (UART3 TX)
```

- The module ships as **either** the PULSE version (pin 1, driven to VCC
  = 9–24 V by a PNP — never wire that to a 3.3 V pin) **or** the AA version
  (pin 2, open-collector NPN, pulls low). Both are conditioned here onto one
  wired-OR net; whichever pin is populated pulls WAKE low on a detection and
  the other floats harmlessly (R3 holds Q1 off, R1 holds WAKE high).
- **WAKE → P11** because the N6 pinout marks P11 as `WKUP3 — connect to
  ground to wakeup`, so a vehicle above the radar's threshold (command 0x04)
  can lift the N6 out of deep sleep. On a board that never sleeps it is just
  a GPIO interrupt.
- UART is 3.3 V on both ends; no shifting. RS-485 A/B (pins 5–6) are
  broken out on the header but unused.
- 3V3 comes *from the N6* only for the pull-up, so the wake line is referenced
  to the N6's own rail (which stays up in deep sleep — `RAW` and 3V3 are
  always-on per the pinout).
- No fuses: the DFR0535's outputs carry their own short-circuit /
  over-current protection.
- D1/R4 is a bench convenience (3 mA). Snip it for deployment.

Board: 40.64 × 30.48 mm (16 × 12 grid units), 4 × Ø2.7 corner holes for
M2.5 on a 38.1 × 22.86 pattern. In the case it hangs component-side-back on
four 6 mm bosses inside the lid, above the N6 (`../case/`).

## Perfboard placement map

Grid: **columns 1–16 left→right, rows 1–12 bottom→top**, one hole = 0.1".
Top view. `dist/index/pcb.png` shows the same map with install silkscreen:
refdes + value, body outlines, pin names on every header, `+` on the
electrolytics and the LED.

| Part | Value / type | Holes (col,row) | Notes |
|---|---|---|---|
| JR | 1×8 male header | c5–c12, r11 | radar J4 order L→R: PULSE AA RX TX B A GND VCC |
| JN | 1×6 male header | c3–c8, r2 | N6 leads L→R: VIN GND 3V3 TX(P4) RX(P5) WAKE(P11) |
| JP | JST-XH 4-pin (or 1×4 header), vertical | c2, r8–r5 | top→bottom: 12V GND 5V GND, from the DFR0535 |
| R1 | 10k | c4 r9 – c7 r9 | 3V3 → WAKE pull-up |
| R2 | 47k | c8 r10 – c11 r10 | PULSE → Q1 base |
| R3 | 100k | c8 r9 – c11 r9 | Q1 base → GND |
| Q1 | 2N3904 TO-92 | c12 r6 (E), c13 r6 (B), c14 r6 (C) | flat face toward you |
| C1 | 100 µF 25 V radial | c14 r10 (+), c15 r10 (−) | 12 V bulk at the radar header |
| C2 | 100 nF ceramic | c11 r10 – c12 r10 | 12 V |
| C3 | 100 µF 10 V radial | c5 r3 (+), c6 r3 (−) | 5 V bulk |
| D1 | 3 mm LED | c11 r3 (+), c12 r3 (−) | power |
| R4 | 1k | c13 r4 – c16 r4 | LED series |
| mount | Ø2.7 | c1 r2, c1 r11, c16 r2, c16 r11 | M2.5 |

Wiring runs (perfboard jumpers, matching the netlist):

- **12V**: JP.12V(c2r8) — JR.VCC(c12r11) — C1+(c14r10) — C2(c11r10)
- **5V**: JP.5V(c2r6) — JN.VIN(c3r2) — C3+(c5r3) — R4.left(c13r4)
- **GND**: JP.GND(c2r7, c2r5) — JR.GND(c11r11) — JN.GND(c4r2) — C1−(c15r10) — C2(c12r10) — C3−(c6r3) — R3.right(c11r9) — Q1.E(c12r6) — D1−(c12r3)
- **3V3**: JN.3V3(c5r2) — R1.left(c4r9)
- **WAKE**: JN.WAKE(c8r2) — R1.right(c7r9) — JR.AA(c6r11) — Q1.C(c14r6)
- **PULSE**: JR.PULSE(c5r11) — R2.left(c8r10)
- **QB**: R2.right(c11r10) — R3.left(c8r9) — Q1.B(c13r6)
- **RADAR_TX**: JR.TX(c8r11) — JN.RX(c7r2)
- **RADAR_RX**: JR.RX(c7r11) — JN.TX(c6r2)
- **LEDA**: R4.right(c16r4) — D1+(c11r3)

## Cables

- **Radar**: the LD2415H test kit's Dupont pigtail (1.25 mm plug on the
  radar, female Dupont ends) pushes straight onto JR. Check the pigtail's
  wire order against the module's silkscreen once — the datasheet's pin table
  is what JR is labelled from.
- **N6**: six male↔female Dupont jumpers, male end into the N6's female
  headers. On the N6 (lens end at the top, USB at the bottom, as on the
  pinout diagram): `VIN` and `GND` are the two lowest pins of the **right**
  outer column, `P11` is second from the top of that column; `3V3` is the
  lowest pin of the **left** outer column, `P4`/`P5` are 4th/5th from the
  top of it. **Verify against the silkscreen before powering** — the
  column order was read off the pinout PNG, not a board.
- **Power**: 4-way JST-XH lead to the DFR0535's screw terminals `OUT3 +/−`
  and `OUT1 +/−`. Set the DFR0535's `OUT3 SET` switch to **12V**.

## BOM

| Qty | Part | Notes |
|---|---|---|
| 1 | 1×8 male header 0.1" | JR |
| 1 | 1×6 male header 0.1" | JN — right-angle if you want the plugs to lie flat in the case |
| 1 | JST-XH 4-pin vertical (B4B-XH-A) | JP; a 1×4 header also fits the holes |
| 1 | 2N3904 (any small NPN) | Q1 |
| 1 | 10k, 1 | 47k, 1 | 100k, 1 | 1k — 1/4 W axial | R1–R4 |
| 2 | 100 µF electrolytic, 2.5 mm lead (25 V for C1) | C1 C3 |
| 1 | 100 nF ceramic, 2.54 mm | C2 |
| 1 | 3 mm LED | D1 |
| 1 | perfboard ≥ 16 × 12 holes (~41 × 31 mm) | |

## Before first power

1. **12 V must never reach the N6.** With JP unplugged, check JN.VIN ↔
   JR.VCC is open and JP.5V ↔ JP.12V is open.
2. Electrolytic `+` toward the left as marked; LED long leg to R4.
3. With only the radar and JP connected: JR.VCC reads 12 V, WAKE reads
   3.3 V idle (needs the N6 plugged in for the pull-up) and drops to ~0 V
   when you wave a hand fast enough to beat the threshold you set.

## Rebuild

```bash
tsci build index.circuit.tsx --pcb-png --schematic-png   # dist/index/
tsci export index.circuit.tsx -f gerbers -o dist/n6_speedcam_carrier_gerbers.zip
```

Runs on the global `tsci` (0.0.2384); no local `node_modules` needed.
