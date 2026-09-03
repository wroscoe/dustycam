# Power: solar panel → 1S LiPo → 12 V radar + 5 V N6

`budget.py` is the runnable calculator (`python3 cameras/n6_speedcam/hardware/power/budget.py`).
Everything below is what it computes plus the reasoning behind the parts.

## Architecture

```
solar panel (9–24 V class, ≤ 20 W) ── PG7 gland ──► DFR0535 SOLAR IN
                                                       │ MPPT 18 V (DIP #4)
                              1S LiPo ◄── BAT IN ──────┤ LTC3652, ≤ 2 A charge
                                                       ├── OUT3 12 V 0.5 A ──► radar VCC (carrier JP)
                                                       └── OUT1  5 V 1.5 A ──► N6 VIN   (carrier JP)
```

One module does the whole job: the **DFRobot DFR0535 Solar Power Manager**
(datasheet: [`dfr0535_solar_power_manager.pdf`](dfr0535_solar_power_manager.pdf))
takes a 7–30 V panel, MPPT-charges a single 3.7 V LiPo at up to 2 A, and
has three always-on regulated outputs — including the **9/12 V 0.5 A rail
the radar needs**, which is what removes any separate boost converter from
the design. Outputs keep running from the battery at night; quiescent draw
is < 3 mA. 78 × 68 mm; it sits on four posts inside the case body, with the
LiPo pouch taped to the back wall behind it.

Why not the N6's own LiPo connector: the N6 charges its cell from USB only,
so a solar system would need a second charger on the same pack. Feeding
`VIN` 5 V from the DFR0535 keeps one charger, one pack, and leaves the N6's
USB-C free for the bench.

**Panel class.** The DFR0535 wants a "9/12/18 V" panel — the nominal-12 V
kind whose open-circuit voltage is 18–22 V (set MPPT to **18 V**). Do not
exceed 30 V open-circuit (two 12 V panels in series will). If yours are the
small **6 V** hobby panels, this module cannot use them; the sibling
DFR0559 (4.4–6 V in, 5 V out only) can, but you would then need a 5→12 V
boost for the radar (Pololu U3V16F12, 0.1" pins, drops onto the carrier
lattice).

## The numbers

Loads at the battery terminals (converter losses included):

| Load | W | Basis |
|---|---|---|
| Radar, always on | 0.69 | ≤ 50 mA @ 12 V (datasheet) / 87 % |
| N6 awake, no WiFi | 0.87 | 150 mA @ 5 V (openmv.io) / 86 % |
| N6 awake, WiFi burst | 2.03 | **estimate** — measure on the real upload |
| N6 deep sleep + charger idle | 0.016 | 1.6 mA @ 3.7 V (openmv.io) + BQ24075 ~1.5 mA |
| DFR0535 + carrier | 0.03 | < 3 mA + LED |

The radar is the always-on cost — 16.5 Wh/day by itself — because it *is*
the trigger. Everything else depends on how much the N6 sleeps:

| Scenario | Avg W | Wh/day | Battery, 1 day | Panel, Dec | Panel, Jun |
|---|---|---|---|---|---|
| A: always on (today's firmware) | 1.71 | 41 | 13.8 Ah | 34 W | 11 W |
| B: radar-wake, busy road (25 % awake) | 1.04 | 25 | 8.4 Ah | 21 W | 7 W |
| C: radar-wake, quiet road (5 % awake) | 0.79 | 19 | 6.4 Ah | 16 W | 5 W |

"Battery, 1 day" = one sunless day at 80 % depth. Panel = nameplate to
refill a day in Teton County peak-sun-hours (2.0 Dec, 6.0 Jun), 80 % panel
derate, 75 % charge efficiency.

**What that means:**

- **Radar-triggered deep sleep is not optional for solar.** Scenario A needs
  a 34 W panel and ~14 Ah of 1S LiPo just to survive a December day; the
  DFR0535 caps out at 20 W anyway. With the wake line (carrier → N6 `P11`)
  and the radar's trigger threshold set, scenario B/C is a **20 W panel and
  a 6–10 Ah 1S pack** — realistic, and 3 days of autonomy in summer.
- Charge current ≤ 2 A, so a pack under 2 Ah needs its own protection PCB
  (most pouches have one). Two pouches in parallel are fine on the two BAT
  IN connectors (JST-PH 2.0 and a 5.08 mm terminal).
- Winter sizing is dominated by the radar; if the road is quiet at night,
  the firmware can drop the radar's trigger duty by powering `OUT3` off on a
  schedule — the DFR0535's outputs are individually switchable, and the
  carrier's 12 V is only the radar.
- The case has room for one **60 × 36 × 8 mm** pouch (~2–3 Ah) against the
  back wall. For 6–10 Ah put the pack in its own box on the pole and run
  the BAT lead through a second gland; the `BAT` envelope in
  `../case/src/n6_speedcam_case_common.py` is the only thing to change if a
  bigger pouch should live inside.

## What to measure first

1. N6 average draw at 5 V over a real capture-and-upload cycle, and in deep
   sleep with `VIN` powered (openmv.io's 1.6 mA figure is via the BAT
   connector; the BQ24075 charger adds ~1.5 mA whenever `VIN` is present).
2. Radar current at 12 V with sensitivity set for the site.
3. Whether the N6 actually wakes on `P11` from `machine.deepsleep()` under
   OpenMV firmware 5.0 — the pinout promises it; the firmware has not been
   tried.

Replace the constants at the top of `budget.py` with what you measure.

## Cold

LiPo must not be charged below 0 °C. The DFR0535 has no temperature
sensor. For a Wyoming winter either accept that the pack only charges on
sunny afternoons above freezing (the case warms in sun), or add a thermistor
cut-off on the SOLAR IN lead, or use LiFePO4/LTO. The N6 and radar
themselves are fine to −20 °C.
