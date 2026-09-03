# ESP32-S3 camera enclosures

3D-printable, splash-resistant cases for the two small ESP32-S3 camera boards,
with every connector opening on the **underside** and a 1/4"-20 tripod socket
on the back.

| Variant | Board | Case outer (board frame) | Deployed W x H x D |
|---|---|---|---|
| `goouuu` | GOOUUU ESP32-S3-CAM (`goouuu1`) | 79.8 x 37.8 x 24.3 | 37.8 x 79.8 x 24.3 |
| `waveshare` | Waveshare ESP32-S3-CAM-GC0308 | 44.8 x 44.8 x 19.8 | 44.8 x 44.8 x 19.8 |

Add 5.0 mm (goouuu) / 12.5 mm (waveshare) for the tripod pad on the back, and
6.0 mm for the visor on the front. Plastic: goouuu 22.5 + 8.3 cm3, waveshare
12.1 + 5.5 cm3 (body + lid).

The generic Amazon ESP32-S3-CAM clone is very probably the same board as the
GOOUUU — sarg's `goouuu-esp32s3cam` record already lists the dual CH340
"TTL" + native "OTG" USB-C layout and carries "ESP32-S3-CAM 40-pin" as an
alias. **Check yours by eye against `ref/goouuu_esp32s3cam_board.step` before
printing**; if it differs it needs its own variant.

## The deployed pose

Both cases are modelled flat in their board's frame and then **stood on the
connector wall** in the field:

```
        GOOUUU                          Waveshare GC0308
   board -X -> DOWN  (2x USB-C)    board -Y -> DOWN  (USB-C + GH1.25 leads)
   board +X -> UP    (antenna)     board +Y -> UP
   board +Z -> FORWARD (the lid is the front; the lens looks out level)
```

That single decision does three things at once: it puts the ports underneath
as asked, it aims the camera forward instead of sideways, and it is what an
outdoor enclosure wants anyway, because a downward-facing opening cannot pool
water or catch falling rain.

This replaces the pose in the published sargineer designs, which laid the case
flat with the tripod boss on a side wall — their own notes flag the
consequence: *"the camera then looks sideways"*.

## Weather strategy

Splash-resistant, **not sealed**. PETG or ASA, not PLA. The rules live in
`cameras/hardware_common/caseskit.py` so they are written once, and after the
simplification pass they are carried by **angles rather than added features**:

- every opening faces down, or sits under the visor;
- the port and roof faces are **chamfered** (1.5 mm). A drop crossing a sharp
  arris wraps it and tracks back underneath; a chamfered arris breaks it off.
  That is the whole job the old proud drip ring did, without a four-box loop
  stuck to the outside;
- the camera window is a **single conical cut**, flaring outward. The flare is
  field-of-view relief, and the sloped wall sheds instead of holding a ring of
  water against the lens;
- the **visor** is one wedge: rain running down the front meets its slope and
  is thrown forward off the tip, clear of the window;
- the lid joint is a **labyrinth lip** (2.5 tall, 1.2 thick, 0.2/side);
- buttons and the LED are **blind** — 0.6 mm and 0.8 mm printed membranes. A
  button behind three layers of PETG still presses; nothing there has to pass
  a plug, so nothing there is a hole;
- the floor openings run down to the cavity floor line, so they drain it.

**Residual risk, stated plainly:** the lid seam is a labyrinth, not a gasket.
Under sustained driven rain it will eventually admit water. The upgrade path
is an O-ring groove in the lid lip — a change to the lip parameters and a
reprint, not a redesign.

**One thing the simplification gave up:** the old drip ring doubled as feet, so
the case could be set on a bench without resting on its connectors. It cannot
any more. Since these mount on a 1/4"-20 tripod — and the tripod pad already
stops the case sitting flat on its back — that was a weak benefit for a
four-box feature, but it is a real loss if you wanted to stand one on a shelf.

## Interface — GOOUUU

Frame: origin PCB plan bottom-left, +X along the long edge, USB-C on the X=0
end, Z=0 PCB bottom, lens +Z. Cavity x -4.5..70.5, y -2..31, z -8.5..11.

| Feature | Spec |
|---|---|
| Board retention | 2x20 header pins drop into grooves in bed rails; grooves 1.5 wide x 6.5 deep at y 1.8 / 27.2, 0.3 entry chamfer. Header plastic seats on the rails -> PCB bottom at Z=0 |
| X stop | USB-end corner blocks, PCB X=0 edge butts at x -0.3; ledges under the PCB corners, top at -0.1 |
| Lid clamp | corner feet 4.0 x 3.4 on the PCB corners; pad 6 x 9 at (53.5, 10) hovering 0.3 over the WROOM can |
| Lid screws | 4x M2 x 8 self-tap, pilot 1.7 x 8, at (67.75, 0.75), (67.75, 28.25), (-2.25, 0.30), (-2.25, 28.70) |
| **USB opening (floor)** | 23.8 wide x 15.5 tall through the X=0 wall, y 2.6..26.4, z -8.5..7.0, 0.6 lead-in. Clears both plug overmolds (12 x 6.5); reaches the cavity floor line so it drains too |
| Face chamfers | 1.5 on the X=0 floor face and the +X roof face |
| Lens window | conical, dia 12.0 inside flaring to dia 16.0 outside, at (46.2, 15.0) |
| Visor | 28 wide, 6.0 proj x 6.0 high wedge starting at x 55.7 |
| Buttons / LED | BOOT + RST dia 6.0 blind, 0.6 membrane; LED dia 3.0 blind, 0.8 membrane |
| Tripod | 1/4"-20 heat-set (ruthex RX-1/4-20) at (33.0, 14.5) on the -Z back face; dia 16 pad **5.0 proud** + dia 16 internal column 8.0 tall; blind pocket dia 8.0 x 13.5 |
| microSD slot | **removed.** The +X wall is the roof once stood up; an open slot there would drink. Card access = take the lid off |

## Interface — Waveshare GC0308

Frame: origin PCB plan bottom-left, Z=0 PCB bottom, USB-C on the Y=0 edge.
Cavity 40 x 40, z -5.0..10.0.

| Feature | Spec |
|---|---|
| Board fixing | 4x M2 x 16 self-tap through lid posts + PCB into body bosses on the 32.6 x 32.6 pattern; 5.0 mm plenum under the board |
| **USB opening (floor)** | 13.0 wide x 7.5 tall through the Y=0 wall, x 12..25, z -5.0..2.5. Overmold spans -4.9..1.6 |
| **Lead slot (floor)** | 10.0 x 4.0 at x 6.0, z -5.0..-1.0 |
| Face chamfers | 1.5 on the Y=0 floor face and the +Y roof face |
| Camera window | conical, dia 18.0 inside flaring to dia 22.0 outside, at (20.0, 24.0) |
| Visor | 34 wide, 6.0 proj x 6.0 high wedge starting at y 36.5 |
| Tripod | 1/4"-20 heat-set at (18.5, 18.5) on the -Z back face; dia 16 pad **12.5 proud**, `inner_h = 0` |

Two deliberate differences from the GOOUUU case, both forced by missing data:

- **The tripod pad is 12.5 mm proud instead of 5.** The GOOUUU gets a short
  pad because an internal column can take most of the insert's length. This
  board's underside is a field of GH1.25/FPC connectors whose positions are
  recorded as *unmeasured*, so nothing is allowed to intrude into the plenum.
  Once the underside is calipered, dropping `TRIPOD_INNER_H` in a column of
  known-clear height shortens the pad — it is one parameter.
- **The lead slot is generous and positioned by convention, not by
  measurement.** The battery/speaker connector XY is unknown, but the leads run
  inside the 5 mm plenum before they reach the wall, so they can arrive at one
  slot from anywhere on the board.

Note the board's battery connector is **GH1.25, not JST-PH** — Adafruit cells
will not plug in without a pigtail. You selected USB-power-only, so neither
case has a battery bay; the Waveshare keeps its connector accessible anyway.

## Layout

```
src/     parametric build123d source. <name>.step.py are thin gen_step()
         entry points; the geometry and every parameter live in the
         *_cam_case_common.py modules. Generated .step sits beside its
         generator (the cad skill regenerates in place).
ref/     board reference models, measured from rather than guessed at,
         + UPSTREAM_*.md, the notes from the sargineer bundles this began as
export/  printable meshes per variant (3MF + STL) -- regenerate, don't edit
renders/ verification snapshots
```

Shared features live in `cameras/hardware_common/` (`pcbkit.py` primitives,
`caseskit.py` weather/mount features). That is a **new cross-camera
convention** — the repo's rule is that each camera owns its `hardware/` — but
two cases already consume it and the alternative is triplicating the weather
rules.

## Regenerate and verify

```bash
CAD=~/.claude/skills/cad
cd cameras/esp32_s3_cam/hardware/case/src
$CAD/.venv/bin/python verify_goouuu.py      # 20 checks, exits non-zero on failure
$CAD/.venv/bin/python verify_waveshare.py   # 16 checks
$CAD/.venv/bin/python $CAD/scripts/gen *.step.py --write
```

build123d is not a repo dependency; generation runs on the `cad` skill's venv
so nothing is installed here.

## Verification report

Run 2026-08-28, after the simplification pass. All checks pass on both
variants; `inspect validate` reports `ok: true, failureCount: 0` for all four
printed parts.

Checked, per variant: body/lid/board interference all 0 mm3; the board shifted
+3 mm in X still clears (GOOUUU — its PCB length is read 60–63 across two
photos); header pins clear their grooves; every lid screw rod misses the
board; USB plug overmold probes reach the ports; the lead slot is an open
path; **every floor opening reaches the cavity floor line**, which is what
replaced the separate drains; the tripod pocket stays blind with >=1.0 mm of
plastic above it and never enters the cavity; button and LED membranes are
solid; the lens column is clear, the window flares outward, and the visor
stays out of the forward view.

**One real defect was found and fixed.** The published Waveshare design has
**8.431 mm3 of body/board interference** — two corner bosses lapping the
underside connector envelope. Fixed at source in
`ref/amz-esp32s3-cam-gc0308.py` by clearing a dia 6 keep-out at each mounting
hole, on the physical argument that a connector cannot sit on a hole you screw
through. That is an **assumption, not a measurement** — confirm it during the
caliper pass.

Three defects were found during the reworks and fixed before export. In the
first pass: the drip ring capped the top edge of the USB opening, and the
brow's drip lip hung 0.5 mm into the camera's forward view.

In the simplification pass, a **drainage error in the first design** surfaced
only because deleting the drain forced the question. Standing the case on its
end makes `z` a *horizontal* axis, so water on the inner floor never flows
toward a z-limited opening — the original 3 x 3 drain sat in one spot of a
floor that drains nowhere, and the USB opening stopped 7.5 mm short of the
floor line. Both floor openings now run down to the floor line and drain it
themselves. The check `USB opening reaches the floor` exists to keep that
true.

## Before you print — the standing caveat

**No dimension on either board has ever been calipered.** The GOOUUU planform
was scaled off photos of a *different vendor's* board using the header pitch
as a ruler, and every height is a DevKitC/WROOM typical. The Waveshare outline
and hole pattern come from a vendor drawing, but its heights, hole diameter
and USB-C centre are estimates. Your own open sarg lesson
`part-pages-need-a-per-dimension-confidence-verification` says exactly this
and ends *"this board is owned; one caliper pass would close it."*

The geometry above is exact; the numbers it consumes are not. Highest-risk
values, in order: header-pin length below the PCB (sets the whole GOOUUU Z
stack), tallest-component heights, USB-C shell centre and protrusion, the
Waveshare underside connector positions, and the lens axis.

Suggested order:
1. Caliper both boards; correct the constants in `ref/goouuu_board_ref.py` and
   `ref/amz-esp32s3-cam-gc0308.py`. This is a parameter edit, not a redesign —
   the cases regenerate from it.
2. Re-run both verify scripts.
3. Print a fit coupon first — the port face plus the retention rails, ~10 mm
   deep — and offer the board to it. That catches an estimated-dimension error
   for a few grams instead of a whole case.
4. Then print the full parts: body back-face-down (the tripod pad wants a
   brim), lid outer-face-down. No supports needed either way.
