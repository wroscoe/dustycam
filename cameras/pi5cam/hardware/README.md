# pi5cam Hardware

Design sources and engineering calculations for the Raspberry Pi camera. The
narrative build walkthroughs live in [`../../../docs/`](../../../docs/); this
tree holds the actual design files and the runnable math they reference.

## Layout

| Directory | Contents |
|---|---|
| [`case/`](case/) | 3D-printable enclosure. Parametric CAD source in `src/`, ready-to-print STEP/STL exports per board variant in `export/`. |
| [`pcb/`](pcb/) | Carrier/power PCB (KiCad project). Fabrication outputs (gerbers, drill, schematic PDF) in `export/`. |
| [`optics/`](optics/) | Camera sensor + lens selection math: field of view, ground coverage, pixels-on-target at distance. |
| [`power/`](power/) | Power budget and battery/solar sizing — both the design doc and the runnable calculator. |

## Bill of Materials

Core build (see [`../../../docs/pi5_build_guide.md`](../../../docs/pi5_build_guide.md)):

| Part | Notes |
|---|---|
| Raspberry Pi 5 | Pi Zero 2 W variant: no guide written yet |
| Raspberry Pi Camera | HQ (IMX477) recommended; Module 3 and Global Shutter also supported — see [`optics/`](optics/) for the tradeoffs |
| C/CS-mount lens | Focal length depends on target distance — compute with `optics/fov.py` |
| microSD card | 16 GB+ |
| 3D printed case | Print from `case/export/<variant>/` |

Optional battery + solar:

| Part | Notes |
|---|---|
| Waveshare UPS HAT (5V/5A) | |
| 18650 Li-ion cell(s) | 3.7 V |
| Solar panel, 12–24 V | 10 W or higher; size with [`power/`](power/) |
| Tunable buck converter | Panel → UPS input |

## Conventions

- Design **source** (parametric CAD `.py`, KiCad files) is the ground truth and
  lives in git. **Exports** (STEP/STL/gerbers) are regenerated artifacts kept
  under `export/` for convenience — regenerate rather than hand-edit them.
- Large binary exports that churn should move to git-lfs or release
  attachments rather than the repo.
- Engineering math is code, not spreadsheets: each calc module is runnable
  (`python3 cameras/pi5cam/hardware/optics/fov.py`) and documents its
  assumptions inline.
