# n6_speedcam enclosure

One printed box for the whole roadside unit: OpenMV N6 + HLK-LD2415H radar
behind a flat front with the lens standing proud, the carrier board, the
DFR0535 solar manager and a 1S LiPo. Every opening is on the underside, a
rain hood covers the front, the front plate is a hinge-in door with no
fasteners on its face, and a 1/4"-20 tripod insert sits in the back.

| | Board frame | Deployed W × H × D |
|---|---|---|
| Outer box | 117.8 × 87.8 × 62.2 | 117.8 wide × 87.8 tall × 62.2 deep |
| + hood | 15.0 forward of the lid face, full width, with a 2.5 lip hanging under it | |
| + lens | 5.0 proud of the lid face | |
| Cavity | 113 × 83 × 57.4 | |
| Plastic | body 86.5 cm³, lid 22.7 cm³ | |

Splash-resistant, **not sealed**; PETG or ASA. Designed 2026-09-02, **not
yet printed**. The N6 model is measured (from OpenMV's own GLB); the radar,
DFR0535 and carrier are envelope models with the assumptions listed under
"Before you print".

## The deployed pose

Same rule as the ESP32-S3 cases: modelled flat in the **N6 board frame**
(origin at the N6's PCB corner, +Y toward its lens, +Z the optical axis) and
stood on the USB edge in the field:

```
board -Y -> DOWN     N6 USB-C, PG7 gland, drain — all through the floor
board +Y -> UP       roof wall runs on 15 mm past the lid as the hood
board +Z -> FORWARD  lid = front: lens window + 1.2 mm radome panel
```

## The front: a hinge-in door

```
              hood ─────────────────────────────────►  (15 mm, chamfered tip)
   roof ══════╗  ╷ lip  (2.5 down, flat underside, drip chamfer)
              ║ ◄╵──────  0.4 pocket  ──────►  lid top edge (chamfered back + front)
        rail ◄╝ (45°, 1.8 seat)
              ║
   side wall  ║   lid plate 2.4, full outer footprint at sides + bottom
              ║   side lip 1.2 × 3.0 inside the wall, 0.2 clear, 3 crush ribs 0.25 proud
              ║
   floor wall ╚═  bottom lip, 3 crush ribs; lid bottom edge flush, 1.5 drip chamfer
                  pry notch 12 × 1.5 in the floor wall's front edge
                  optional 2 × M2.5 × 8 up through the floor into blocks on the lid back
```

Closing: hold the door tilted a few degrees (bottom out), push its top edge up
into the pocket under the hood until it stops against the roof, rotate the
bottom in. The nine crush ribs (0.05/side net interference, 45° noses) bite
into the cavity walls and hold it. Opening: nail into the pry notch on the
underside, pull the bottom edge out, drop the top edge out of the pocket.
The two floor screws are optional — friction holds it on the bench, add
them for a roadside install.

The lens tip stands 5 mm out of the face (`LENS_PROUD`), so the lock ring
stays behind the lid and the barrel passes a Ø15 window with a 0.6 flare.

## Who carries what

```
LID (front, prints outer-face-down, flat outside)
  radar        4 x M2.5 x 6 into Ø6 bosses, 5 mm off the floor of a 55 x 71 pocket
               thinned to RADOME_T = 1.2 from the inside; antenna face at z 20.05
  N6           2 x M2.5 x 8 from behind, through its Ø2.80 mounting holes, into
               Ø5.5 bosses 22.55 tall (Ø4.4 tip below z 6.5 to clear the camera arm);
               two 4 x 4 anti-bow pads hover 0.3 over the PCB at the USB end
  carrier      4 x M2.5 x 6 into Ø5.5 bosses 6 mm tall, above the N6, parts facing back
  lens window  Ø15 with a 0.6 outward flare; lens tip 5.0 proud
  lips + ribs  side and bottom lips inside the cavity mouth, 9 crush ribs
BODY (back, prints back-face-down, flat outside)
  DFR0535      4 x M3 x 8 into Ø6 posts 16 mm off the back wall, parts facing forward,
               rotated so BAT/SOLAR terminals face the gland side (right)
  LiPo pouch   60 x 36 x 8 envelope taped to the back wall behind the DFR0535
  tripod       ruthex RX-1/4-20 in a blind Ø8 x 13.5 pocket inside a Ø16 x 12.2 column
               at (60, 62) — the back face stays flat
  pocket       lip hanging 2.5 under the hood in front of the lid, 45° rail behind it
  floor        USB-C opening 14 x 11.5 (x 16.7..30.7, z -3.5..8);
               drain 8 x 3 at the cavity floor line (x 60..68);
               PG7 gland bore Ø12.5 at (87, -15); pry notch; 2 optional screw holes
  hood         roof wall extended 15 mm past the lid face, 1.5 chamfer on its tip
```

Z stack, back to front (all checked to nest by `src/verify_n6_speedcam.py`):
back wall −33.5 │ battery to −25.5 │ DFR0535 posts to −17.5, parts to −4.9
│ N6 tails −3.0, PCB 0…1.3, headers to 9.8 │ carrier plugs 2.3…16.3, PCB to
17.85 │ radar J4 to 9.05, antenna face 20.05 │ bosses │ radome 25.05…26.25
│ lens tip 31.25.

## Weather — where the water goes

caseskit's rules, written once in `cameras/hardware_common/caseskit.py`:
every opening faces down (USB, gland, drain, pry notch); floor and roof
arrises are chamfered 1.5 so drops break off instead of tracking under.
On the front, traced path by path:

- **Rain on the hood** runs to its chamfered tip and drips 15 mm clear of
  the face.
- **Rain on the face** runs straight down. There is nothing on the face to
  hold it — no screw heads, no counterbores. It meets the lens barrel,
  parts around it (the window's 0.6 flare is an arris, not a cup), and
  leaves off the lid's chamfered bottom edge.
- **The top seam** is the only one that faces the sky, and it is under the
  hood *and* behind the pocket lip; the lip's own lower arris is chamfered
  so it drips rather than feeds the seam.
- **The side seams** are flush lines on the side faces. A film that wicks
  through runs down the 0.2 gap between side lip and wall and out through
  the underside seam.
- **The bottom seam** faces down; gravity owns it.
- Anything that still gets past a lip lands on the cavity floor and leaves
  by the drain slot at the floor line.

The **radome** is the one feature the ESP32 cases do not have. The lid is
2.4 thick everywhere except a 55 × 71 pocket cut from the inside to 1.2 over
the radar. At 24 GHz in PETG (εr ≈ 2.7) the in-material wavelength is
~7.6 mm; a panel far thinner than that (≤ ~1.2) or a half-wave (~3.8)
transmits, anything between reflects a growing share back into the antenna.
1.2 mm is six 0.2 layers — the thinnest that still prints as a weather wall.
`RADOME_T` is one parameter; 3.8 is the other legitimate value. No paint,
foil or metal on that panel. The hood keeps rain film off it, which matters:
a wet radome is the main outdoor loss at 24 GHz.

**Residual risk:** lips, not gaskets. Sustained driven rain will get in; the
drain is there for that. An O-ring in the top pocket and a foam strip behind
the lips is the upgrade.

## Assembly order

1. Heat-set the 1/4"-20 insert into the body's back column from the outside.
2. Lid, face down on the bench: screw the radar to its 4 bosses (M2.5 × 6,
   heads on the radar's back; the pigtail edge toward the floor side), the
   carrier to its 4 bosses (M2.5 × 6, parts facing you), then the N6 to its
   2 bosses (M2.5 × 8 from the back of the PCB, lens through the window).
3. Body: DFR0535 on its 4 posts (M3 × 8), MPPT switch to 18 V, OUT3 switch
   to 12 V. Tape the pouch to the back wall first — it lives under the
   board. Fit the PG7 gland; solar lead in, to SOLAR IN.
4. Plug: radar pigtail → carrier JR; six Dupont leads carrier JN → N6;
   XH lead carrier JP → DFR0535 OUT3/OUT1; pouch → BAT IN.
5. Close: tilt the door, top edge up into the pocket, swing the bottom in
   until the ribs bite. Optional: 2 × M2.5 × 8 from below.

SD card and the N6's two edge buttons are inside; the USB-C reaches out the
floor for programming. The N6 has to come off the lid to change the SD.

## Layout

```
src/     n6_speedcam_case_common.py  every parameter + body()/lid() builders
         *.step.py                   thin gen_step() entries: body, lid,
                                     assembly (board frame), assembly_zup
                                     (deployed pose, for the viewer)
         verify_n6_speedcam.py       62 fit checks, exits non-zero on failure
ref/     envelope/reference models + OPENMV_N6_DIMENSIONS.md (measured)
export/  n6_speedcam_case_{body,lid}.{stl,3mf}  — regenerate, don't edit
renders/ review snapshots (gitignored *.png; regenerate with the cad skill)
```

## Regenerate and verify

```bash
CAD=~/.claude/skills/cad
cd cameras/n6_speedcam/hardware/case/src
$CAD/.venv/bin/python verify_n6_speedcam.py
$CAD/.venv/bin/python $CAD/scripts/gen *.step.py --write
$CAD/.venv/bin/python $CAD/scripts/export n6_speedcam_case_body.step.py --stl ../export/n6_speedcam_case_body.stl --3mf ../export/n6_speedcam_case_body.3mf
$CAD/.venv/bin/python $CAD/scripts/export n6_speedcam_case_lid.step.py  --stl ../export/n6_speedcam_case_lid.stl  --3mf ../export/n6_speedcam_case_lid.3mf
```

## Verification report (2026-09-02, hinge-in door revision)

All 83 checks pass; `inspect validate` reports `ok: true, failureCount: 0`
for both printed parts, each a single solid.

Checked: body and lid against each of the five reference parts (N6, radar,
DFR0535, carrier, battery envelope) — all 0 mm³; the five parts against each
other — all 0; lid-without-ribs against body — 0, and lid against body
equals exactly the crush-rib volume (7.46 mm³, designed); the door rotates
in from 2°, 4° and 6° open with the N6, radar and carrier riding on it,
clearing the body and the DFR0535; pushed 1 mm forward it is stopped by the
pocket lip, pushed 1 mm back by the rail; the top edge sits 0.2 under the
roof and the bottom edge is flush with the floor face; the lens tip is
exactly 5.0 proud and the lock ring stays ≥ 0.5 behind the lid; a 12 × 6.5
USB-C overmold passes the floor opening; the drain reaches the cavity floor
line; a Ø12.3 gland probe and a Ø18 × 8 nut envelope inside clear
everything; the tripod pocket takes the Ø8 × 13.5 insert, stays blind with
≥ 1 mm of plastic, its column clears the battery, and the back face is
flat; the radome pocket holds only a 1.2 mm membrane plus the four bosses;
the Ø14.9 lens column is clear; every screw rod (N6 ×2, radar ×4, DFR0535
×4, optional floor ×2) passes only its hole; the carrier sits wholly above
the N6; the DFR0535's parts stay ≥ 1 mm behind the N6's header tails; the
lid's outer face is a single flat face.

Caught by the checks during the reworks: the lid plate ran under the hood
(669 mm³); the roof chamfer failed after the hood union; a sloped underside
on the pocket lip let the door cam straight out of the pocket (retention
check read 0 — it is now flat); and the door's top-front corner touches the
roof if you try to swing it in from more than ~8° open, so the checks pin
the working range at ≤ 6° (which needs the bottom only ~9 mm out). Found in
the viewer rather than by a check: the hood was unioned onto the roof *after*
the roof's 1.5 chamfer, so the whole 15 mm hood hung off a 0.9 mm neck with
a groove behind it (face `o1.1.f59`). A second attempt rooted it 1.5 lower
but the union still left a 22.5° groove at the seam (face `o1.1.f35`). The
hood is now not a union at all: the body is one extrusion to the hood tip,
chamfered while every face is still a rectangle, with everything in front of
the lid plane except the roof strip cut away afterwards. The roof surface is
flat at 82.4 from the back chamfer to the tip chamfer, and two checks (probe
volume at the root, no exposed face behind the hood) keep it that way.

## Before you print — what is assumed, not measured

- **LD2415H**: 69 × 53 × 5 and the hole pattern are from the datasheet; PCB
  thickness (1.6) and the back-side part height (3.4) are guesses. The **J4
  connector's footprint is measured** (14.8 wide, 7.3 in from the short edge,
  8.2 from the back-view left edge); its depth and height (6 × 6) are not.
  It now sits inside the board outline, so the 11 mm between the radar's
  edge and the floor is pigtail room, not connector room.
- **DFR0535**: 78 × 68 is documented; the corner hole positions (3.5 in,
  Ø3.2), component height (11) and heatsink position are not. If the holes
  differ, `ref/dfr0535_ref.py:HOLES` is the only edit.
- **N6**: PCB thickness 1.30 and header tail length 3.0 are from the vendor
  model; both feed the boss length. The pinout-derived assumption that the
  two Ø2.80 holes are clear of parts on the back is what lets M2.5 heads sit
  there.
- **Carrier**: Dupont plugs modelled 14 mm tall; right-angle headers make
  that moot.
- **Battery**: 60 × 36 × 8 is a placeholder for whatever pouch you have.
  `BAT` in the common module; the bay behind the DFR0535 is 16 mm deep less
  the 1.6 board and the heatsink, so up to ~10 thick fits.
- The gland bore is a horizontal Ø12.5 hole in a vertical wall — bridged,
  drill it clean. The pocket lip under the hood is a 2.5 mm ledge facing
  the bed when the body prints back-face-down — the one feature that may
  want a line of support.
- Crush-rib interference (0.05/side) is the mywarehouse tolerances value
  for this printer; if the door is loose, raise `RIB_PROUD`; if it will not
  seat, lower it.
