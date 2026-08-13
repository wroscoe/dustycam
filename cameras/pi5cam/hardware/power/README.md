# Power Management

Power budget, battery/solar sizing, and the 5 V delivery design for the Pi 5
build.

- `budget.py` — runnable calculator: duty-cycle budget, peak rail sizing,
  battery-side currents, and 5 V wire-drop limits.
  `python3 cameras/pi5cam/hardware/power/budget.py`

## Architecture: separate battery pack

**Decision:** the solar/battery system is a standalone box, separate from the
camera. It outputs the 12–24 V bus over a single cable; the camera case
contains only the buck converter and the Pi. Rationale:

- The pack is reusable across projects (camera, rover, sensors) — it's a
  generic "solar power brick" with a 12–24 V output.
- The camera case stays small — no cells, charger, or UPS inside.
- It matches the distribution rule below: only low-current high-voltage
  crosses the cable, so cable length and connector choice stop mattering.

## How much power the system actually needs

Three different numbers, used for different decisions (Pi 5 + HQ cam, no USB
peripherals):

| Number | Value | Sizes what |
|---|---|---|
| Average draw @ 25% duty | ~4.3 W | battery capacity, solar panel |
| Realistic peak (all cores + ISP, ×1.25 margin) | ~17.5 W ≈ 3.4 A @ 5.1 V | regulator, wiring, fuse |
| Official spec | 25 W (5.1 V / 5 A) | only if USB peripherals are added |

Key insight: **a headless DustyCam never draws 25 W.** A Pi 5 running
flat-out measures ~12–13 W; the rest of the official 5 A budget is headroom
for downstream USB devices (SSDs, modems). We design the rail for 5 A anyway
— the cost delta on a buck converter is trivial and it removes the rail as a
constraint — but battery/solar sizing should use the duty-cycle average, not
the spec.

### The 5 A negotiation quirk

The Pi 5 decides its USB current limit by USB-PD negotiation. A supply that
*delivers* 5 A but doesn't *negotiate* it (any dumb buck feeding USB-C) makes
the Pi cap its USB ports at 600 mA total and log a max-current warning — CPU
performance is unaffected. Since this build has no USB peripherals that's
cosmetic, but silence it by telling the firmware the supply is good:

```
# /boot/firmware/config.txt
usb_max_current_enable=1
```

(older firmware: `rpi-eeprom-config` → `PSU_MAX_CURRENT=5000`.)

## Delivering 5 V / 5 A: the rule

**Never distribute 5 V. Distribute 12–24 V and buck to 5.1 V within ~30 cm of
the Pi.**

The Pi 5 browns out below ~4.75 V. With the regulator set to 5.1 V and
150 mV reserved for transients, the entire wiring + connector drop budget at
5 A is **0.20 V = 40 mΩ round trip**. `budget.py` shows what that buys:
30 cm of 20 AWG plus one ordinary connector uses the whole budget. At 24 V
the same 17.5 W is only 0.7 A — drop becomes a non-issue even over a
several-meter pack-to-camera cable, and it resolves the barrel-connector
question from `docs/build_guide.md`: a standard 5.5×2.1 mm barrel (rated
5 A) on the 24 V input is loafing.

### Power chain

```
┌─ battery pack box (shared across projects) ─────┐
│ solar 12–24 V ─► charge controller ─► 18650 pack │
│                       │                          │
│                  temp sensor                     │
│               (charge inhibit <0°C)              │
└──────────────────┬──────────────────────────────┘
                   │ 12–24 V bus, one cable, barrel connector
┌─ camera case ────┼──────────────────────────────┐
│             buck ≥6 A cont., 5.1 V, ≤30 cm run  │
│                   │                              │
│        Pi 5 (USB-C or 5V GPIO pins) + camera     │
└─────────────────────────────────────────────────┘
```

Injection point at the Pi, pick one:

1. **USB-C connector** (preferred): keeps the Pi's input protection in
   circuit. Cut a USB-C cable short or use a USB-C plug breakout; set
   `usb_max_current_enable=1`.
2. **GPIO 5 V pins (2+4 / 6+9)**: bypasses the input fuse and protection —
   add a 5 A fuse and check polarity twice. Use both 5 V and both GND
   pins to halve connector-pin current.

Buck converter requirements: rated ≥6 A *continuous* (not peak), set-point
adjustable to 5.1–5.2 V, current limit above the Pi's boot inrush.

### Battery-side reality check

Boosting/regulating from the pack multiplies cell current. From `budget.py`:
at the 17.5 W design peak with nearly-flat cells (3.2 V), the pack must
source **~6 A**; a 25 W worst case needs **~9 A**. A single 18650 rated 5 A
can't do it — use ≥2 cells in parallel (2P) or high-drain cells, and verify
the charge controller's output stage is rated for sustained (not peak) load.

## Cold weather

Li-ion cells are damaged by charging below 0 °C (discharge is fine to about
−20 °C). What products do about it, roughly in order of effort:

1. **Charge inhibit** — BMS blocks charging below 0 °C and tapers current
   between 0–10 °C (the JEITA curve). Camera keeps running through cold
   snaps; it just can't recharge until the pack warms. Fails only in
   extended cold: pack drains with no recharge.
2. **Heated packs** — resistive film around the cells; when charge current
   arrives and the pack is cold, the BMS diverts that current to the heater,
   then switches to charging once above ~0 °C. Crucial detail: **the heater
   runs off the charge source (solar), never the battery** — a heater on
   battery power in deep cold is a death spiral (5–10 W pad vs. our 4.3 W
   camera load).
3. **Chemistry swap** — LTO charges at −30 °C (pricey, low density); AGM
   lead-acid charges below freezing at reduced rate (heavy, half the cycles,
   zero thermal management — the classic remote-telemetry choice).
4. **Insulation + waste heat** — co-locating electronics with the pack uses
   their dissipation as a free heater. With the pack in its own box (see
   Architecture) this mostly doesn't apply; insulation still slows the
   overnight temperature swing and lets discharge self-heating help.

**Decision (v1):** charge inhibit only — the pack's charge controller must
have a temp probe on the cells and block charging below 0 °C. Accept
possible downtime in extended deep cold. **Later:** add a solar-surplus
heater pad + thermostat (option 2) to the pack box for multi-day cold
recovery; the box should leave room for it.

## Reference parts (researched 2026-07, prices approximate)

| Part | Pick | ~Price | Why |
|---|---|---|---|
| Charge controller | Victron SmartSolar MPPT 75/10 | $65–77 | MPPT, LiFePO4 profile, 10 A **load output with configurable low-voltage disconnect** (wire the camera here — free over-discharge protection), Bluetooth monitoring, 5-yr warranty |
| Temp cutoff | Victron Smart Battery Sense | ~$35 | Straps to the battery; enables the controller's lithium low-temp charge cutoff via VE.Smart. Required: the 75/10 canNOT do low-temp cutoff from its internal sensor, and small LiFePO4 batteries rarely have it in their BMS |
| Battery | 12 V 20 Ah LiFePO4 (LiTime / Redodo / Power Queen) | $80–110 | 256 Wh × 0.8 usable ≈ **2 days autonomy** at 4.3 W avg. Output is the 12 V bus directly — no boost stage. If the chosen model's BMS has its own low-temp cutoff, that's belt-and-suspenders |
| Panel | 100 W 12 V mono (Renogy/Newpowa; 50 W is the spec minimum at ~$75) | $90–110 | Calc says ~34 W sustains the average; 100 W buys Wyoming-winter margin. Checks: Voc ~24 V < 75 V limit; 100 W/12.8 V ≈ 7.8 A < 10 A limit |

System notes:
- Camera bus = controller **load output** (~12.8 V) → barrel cable → buck in
  camera case. Peak 17.5 W ≈ 1.4 A on the bus — any cable works.
- Mount the panel steep (70–80°) for winter: sheds snow, catches low sun.
- Total pack cost ≈ **$270–330** including panel.

## Open items

- [ ] Measure real draw with a USB power meter: idle, capture, sustained
      inference. The 25% duty cycle in `budget.py` is a guess and dominates
      battery/solar sizing.
- [ ] Order + bench-test the reference parts above; confirm the Smart Battery
      Sense low-temp cutoff actually blocks charging (chill test).
- [ ] Winter check: `SOLAR_DERATE = 0.15` is generic; Wyoming winter sun +
      snow cover may need a bigger panel or accepting downtime.
- [ ] Pack box: reserve space + wiring for the future heater pad.
