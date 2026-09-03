# OpenMV Cam RT1062 (R6) — drip-resistant two-part printed case

Outdoor-ish camera housing for the current RT1062 board revision (R6, the one
the official "V5/V6 case" is for). Lens looks horizontally, the USB-C and
battery connectors hang out of an **open bottom**, and a **1/4-20 heat-set
insert** in the back plate takes any tripod / ball-head / wall-bracket stud.
Everything is parametric build123d (`caselib.py`); the board envelope it is
built around is in `DIMENSIONS.md` (measured from OpenMV's own R6 model).

Overall **42.2 W × 58.4 H × 52.3 D** mm (hood tip to roof lip), lens stands
8 mm proud of the front face inside a 12 mm hood. Cup 24.7 cm³, plate 7.7 cm³.

## Parts

| Part | Generator | Print file | Orientation | Supports |
|---|---|---|---|---|
| Cup (front, sides, roof, hood, corner blocks) | `cup.step.py` | `cup-print.stl` | standing on its open bottom | none |
| Back plate (board posts, insert boss, rim, screw holes) | `plate.step.py` | `plate-print.stl` | flat, outer face on the bed | none |

`assembly.step.py` / `assembly-zup.step.py` = cup + plate + simplified board
mock for review (the Z-up one matches the viewer's up axis). `board-mock.step.py`
is the R6 envelope used for the interference checks in `check_fit.py`.

## How it goes together

```
1. Heat-set the 1/4-20 insert into the plate from the OUTSIDE (Ø8.8 hole, 13 deep).
2. Remove the two M2 x 4 camera-module screws from the board.
3. Sit the board on the plate: the two Ø5 posts land under the board's mounting
   holes, the two Ø3.4 posts rest under the bottom corners.
4. Fasten with 2 x M2 x 12 pan-head through camera module -> M2 SMT spacer ->
   PCB -> post (Ø1.7 pilot, self-tapping, 6 mm engagement).
5. Slide the plate+board straight into the cup from the BACK; the lens passes
   through the Ø15 hole, the SD card rides in the groove in the right wall,
   the U-rim on the plate locates in the cavity.
6. 4 x M2 x 10 self-tapping through the plate's counterbores into the cup's
   corner blocks (they sit below the board's bottom edge and above its top edge).
```

Water: flat roof with 2 mm drip chamfers, roof overhangs the back seam by
2 mm, front seam is the lens hole under the hood, all other seams face
sideways or down, open bottom drains. Not sealed — drip/rain-resistant only.

Switch access: the two side-actuated edge switches on the board's top edge sit
under two Ø7 chimneys on the roof; push a paperclip (Ø2.6 hole) straight down.
The chimneys keep pooled water off the holes.

Battery: the bay behind the board is 10.6 mm deep, open to the bottom, clear
of the insert boss up to Y = 34 — a 5 × 25 × 35 mm 1S LiPo (502535) fits with
its lead going straight to the JST at the bottom edge. Nothing retains it;
use foam tape.

## Key parameters (caselib.py)

| Name | Value | Why |
|---|---|---|
| `CLR` | 0.30 | board edge to cavity wall |
| `WALL` / `FRONT_T` / `ROOF_T` / `PLATE_T` | 3.0 / 2.4 / 2.4 / 2.4 | |
| `Z_FRONT_IN` | 19.5 | 0.55 above the M12 holder top |
| `LENS_HOLE_D` | 16.0 | knurled focus ring is ~Ø15.5 on the real lens (rev B, first print bound at Ø15) |
| `HOOD_L`, `HOOD_R_IN`, `HOOD_FLAT_IN` | 12, 10.5, 8.5 | tip 3.95 mm past the glass; no vignette to 40° half-angle (flat ceiling bridges 12.3 mm) |
| `Z_PLATE_IN` | −14.0 | battery bay + room for the inward insert boss |
| `INSERT_HOLE_D`, `INSERT_HOLE_DEPTH` | 8.8, 13.0 | for a 9.5 OD × 12.7 brass 1/4-20 insert — **check your insert's datasheet** (OD − 0.3 … 0.5) |
| `M2_PILOT_D` | 1.70 | self-tapping M2 in PLA/PETG |
| `Y_CEIL` | 50.0 | switch tips at 45.11 + 4.5 mm top blocks |
| `SD_POCKET_*` | 1.5 deep groove | card can stay in during slide-in |

Assumes **no 2×8 headers soldered** (they would need ~10 mm more depth both
sides). Lens focus ring is reachable inside the hood; pull the plate out for
big adjustments.

## Printability

- Cup: bottom edge + corner blocks are the bed contact; walls vertical; roof is
  a 25.5 mm bridge between the top blocks; hood ceiling is a 12.3 mm bridge;
  hood underside is cut back at 45°; SD groove and switch chimneys vertical;
  lens hole is a Ø16 horizontal hole (fine on a 0.4 nozzle); roof drip lip
  overhangs 4.4 mm at 90° (2.4 thick, trivial). Use a brim if your bed
  adhesion is marginal — the part is 58 mm tall on a 42 × 36 footprint.
- Plate: everything grows up from a flat outer face; counterbores and the
  insert bore start on the bed. 12.6 mm tall Ø5 posts are fine unsupported.
- PETG recommended outdoors; PLA sags in a hot enclosure in the sun.

## Validation that ran (2026-09-01)

- `scripts/inspect validate` cup.step / plate.step: ok, 1 solid each.
- `check_fit.py`: cup∩board 0, plate∩board 0, cup∩plate 0, cup∩USB-plug
  envelope 0, cup∩board over the full 40 mm slide-in path 0, nothing in the
  cable zone below the board.
- Snapshots `snap-*.png` reviewed: hood, chimneys, lip, blocks, posts, rim all
  present and where intended.

## Reference designs (`ref/`)

- `RT1062_R6.glb` — OpenMV's R6 board model (source of every dimension).
- `Case-rt1062-V4.glb` — OpenMV's official V4 case: lid + base, board on two
  14 mm hex standoffs with long screws through the camera-module holes
  (the fastening idea this design reuses), 400 mAh 502535 LiPo under the board.
- Community: lirex's bendy-tripod case (thingiverse.com/thing:6779367,
  CC BY-NC-SA, one 3MF) did not download through the automated browser; grab it
  by hand if wanted.
- Official V5/V6 case and board STEP are on GrabCAD (login needed):
  grabcad.com/library/openmv-cam-rt1062-v5-v6-case-1, .../openmv-cam-rt1062-1.
- ErikEngineer's press-lid case (printables.com/model/798382) needs a Printables login.

## Regenerate

```bash
SK=~/.claude/skills/cad
$SK/.venv/bin/python check_fit.py
$SK/.venv/bin/python $SK/scripts/gen cup.step.py plate.step.py assembly-zup.step.py cup-print.step.py plate-print.step.py --write
$SK/.venv/bin/python $SK/scripts/export cup-print.step.py --stl
$SK/.venv/bin/python $SK/scripts/export plate-print.step.py --stl
```
