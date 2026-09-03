# n6_speedcam hardware

Design sources for the radar speed camera: an OpenMV N6 paired with a
Hi-Link HLK-LD2415H 24 GHz vehicle-speed radar, solar powered. Every number
the tree relies on is written down next to where it came from.

| Directory | Contents |
|---|---|
| [`radar/`](radar/) | The LD2415H: pinout, the serial protocol (ASCII speed frames + `CF` config commands), mechanical data, and the datasheet PDF. |
| [`carrier/`](carrier/) | tscircuit board that joins radar, N6 and solar manager: 12 V/5 V rails, trigger→wake conditioning, UART. Perfboard-first; gerbers in `dist/`. |
| [`power/`](power/) | Solar → 1S LiPo → 12 V + 5 V design around the DFRobot DFR0535, and `budget.py`, the runnable panel/battery sizing. |
| [`case/`](case/) | Weather-resistant printed enclosure (build123d): lid carries the sensors behind a flat radome, body carries the power; USB, gland and drain on the underside; tripod insert on the back. |

## How it hangs together

```
                    ┌──────────── lid (front) ────────────┐
 road  <── 24 GHz ──┤ LD2415H ─ J4 pigtail ─┐              │
 road  <── lens  ───┤ OpenMV N6 ◄─ 6 Dupont ┤ carrier      │
                    └───────────────────────┴──────┬───────┘
                                              XH 4-way: 12 V, 5 V
                    ┌──────────── body (back) ─────┴───────┐
 panel ── gland ────┤ DFR0535 (MPPT 18 V, OUT3 12 V, OUT1 5 V) ── 1S LiPo │
                    └──────────────────────────────────────┘
```

Signal path: radar streams `V+012.3\r\n` at 9600 baud into the N6's UART3
(P4/P5); the radar's detection output, conditioned on the carrier, pulls the
N6's `P11 / WKUP3` low so the camera can deep-sleep between vehicles. That
wake line is what makes solar viable — see `power/README.md`.

## Conventions

As [`../../pi5cam/hardware/`](../../pi5cam/hardware/): design **source**
(tscircuit `.tsx`, build123d `.py`) is the ground truth and lives in git;
**exports** (gerbers, STEP, STL/3MF) are regenerated under `dist/` or
`export/`. Engineering math is runnable code (`power/budget.py`). Case
features shared with the other cameras come from
[`../../hardware_common/`](../../hardware_common/).

## Status (2026-09-02)

Everything here is designed, machine-checked and **unbuilt**: the carrier
netlist passes tscircuit's checks, the case passes 62 boolean fit checks
against reference models, the radar driver passes 20 host tests. No part
has been powered, printed or calipered. The list of assumptions to retire
first is at the end of each subdirectory's README; the case's is the longest.
