# OpenMV Cam N6 + 1S LiPo — 3-part printed case

Overall **47.21 × 60.88 × 42.30 mm**, lens barrel stands **5.35 mm proud** of
the front face. All mm. Frame is the N6 board frame from
`ref/DIMENSIONS.md` (origin = PCB bottom-left corner on the PCB
bottom face, +Y toward the lens, +Z along the optical axis).

## Parts

| Part | File | Print orientation | Volume |
|---|---|---|---|
| Front cup | `front-cup.step` / `.stl` | **Face down** (lens face on the bed) | 28.1 cm³ |
| Cam plate | `cam-plate.step` / `.stl` | **Flat, bosses up** | 6.5 cm³ |
| Back cup | `back-cup.step` / `.stl` | **Back face down** | 8.5 cm³ |

`n6-case.step` is the assembled view. `fitcheck.step` is a review-only cutaway
with the real N6 model and the battery mock inside it.

No supports on any part. Largest unsupported feature is the 1.20 shoulder
ledge in the front cup and the 1.00 bay ledge in the back cup — both print as
short cantilevers on a 0.4/0.2 profile.

## How it goes together

```
front cup  ── straight bore over the board, 1.20 internal shoulder at z = -3.50
cam plate  ── seats on that shoulder; N6 bolts to it with 2 × M2.5 × 6
back cup   ── lip presses into the front cup socket and pushes the plate up
              against the shoulder; encloses the battery bay
```

1. Bolt the N6 to the cam plate: **2 × M2.5 × 6 pan head**, from the component
   side, through the board's two Ø2.80 gold-ringed mounting holes at
   (3.048, 41.402) and (32.512, 41.402), into the Ø2.10 pilots in the plate
   bosses. 4.70 mm thread engagement — do not use a longer screw, the pilot
   is blind at 5.30 with a 1.20 floor under it.
2. Feed the battery leads down through the wire slot in the plate
   (x 6–15, y −3.9…−1.9) and plug into the JST at the USB end. Tape the pouch
   to the back face of the plate.
3. Drop board + plate into the front cup. The plate lands on the shoulder.
4. Press the back cup lip into the socket until its flange butts the front cup
   rim. It is a friction fit on crush ribs — firm thumb pressure, no tools.

Disassembly: grip the 2.4 mm seam band at the back and pull the back cup off;
the plate and board then lift straight out.

## Rev D (2026-09-02)

The rev C microSD through-slot only worked for inserting the card AFTER
assembly; sliding the board in with a card seated still collided with the
socket-bore wall. Requirement clarified: the card goes in once, before the
camera does, and needs no outside access. So the slot is gone and the +X bore
side now runs **3.60 wide for its full height** (`CARD_CH`, vs 0.60
`BOARD_CLR`) — a card channel like the wire channel. Case is 3.0 wider:
47.21 × 60.88. The wall is fully closed again on that side. Validated with a
40-step slide-in sweep of board + plate + seated card: zero contact over the
full 20 mm of insertion travel.

## Rev C (2026-09-02)

Three usability fixes after review:

- **microSD slot** through the +X wall (13.0 × 3.6) — superseded by rev D.
  A seated card's tail reaches 3.06 mm past the PCB edge — rev B could not
  physically close with a card inserted. The card now passes through the wall
  and can be swapped from outside (validated with a full slide-out sweep).
- **Battery-cable room**: the -Y wire channel doubled 4.0 → 8.0 mm
  (`WIRE_CH`), case grows 4 mm in Y. The plate wire slot grew to 11 × 5.5 and
  now passes a mated JST-PH plug, so the battery never needs to be unplugged
  to disassemble.
- **USB-C port** through the -Y wall: 15.0 × 9.5 (r 2.5), centred on the
  receptacle, sized to pass a plug overmold up to 12 × 7. Three of the five
  vents made way for it; the port's own open area exceeds what they provided.

New reference mocks (`sd_card_mock`, `usb_plug_mock`, `battery_cable_mock`)
ride along in `fitcheck.step.py` and the interference validation.

## Rev B (2026-09-02)

Lens aperture opened **Ø15.00 → Ø16.00** (`LENS_HOLE_D`): print 1 of the
RT1062 case (same M12 lens family) bound on the knurled focus ring, ~Ø15.5
actual vs the Ø14.0 barrel in OpenMV's GLB. Ø16.00 clears the knurl 0.25/side
and the Ø16.20 lock ring stays captive behind the face (0.10/side). Fit
constants and `edge_crush_rib()` now live in the project-local `fits.py`
(mywarehouse retired 2026-08); rib flanks are functionally identical but
measure 15.44 mm³ total crush instead of 16.85.

## Fits (from project-local `fits.py`, ex-mywarehouse)

| Interface | Value |
|---|---|
| Back cup lip → front cup socket | 0.15/side clearance |
| Crush ribs on the lip | 6 × 6.0 long, 0.25 proud → 0.10/side net crush (15.44 mm³ total) |
| Lip engagement | 7.50 |
| Cam plate → socket | 0.20/side |
| Plate bearing on the shoulder | 1.00/side |
| Board → front cup bore | 0.60/side (0.70 at the PCB corners) |
| Lens barrel Ø14.00 → aperture Ø16.00 | 1.00/side (0.25/side on the ~Ø15.5 focus knurl) |
| Lead-in chamfers | 0.60 × 45° on the socket mouth and the lip nose |
| Elephant-foot chamfer | 0.40 × 45° on every bed-contact perimeter |

Walls 2.40 (6 perimeters); lip wall 1.60 (4 perimeters, per the
friction-feature rule); plate 3.00.

## Battery bay

**35.91 × 52.58 × 7.50**, sized for a 30 × 40 × 6 pouch — 6.00 max thickness
plus the 1.50 swell/wire allowance from `hardware/batteries.md`. Nothing rigid
bears on the pouch face; it sits between the plate back face and the back cup
floor. A bigger pack fits: anything up to roughly 35 × 48 × 6 will drop in.

## Vents

Two 2.40 × 12.00 stadium slots through the bottom (−Y, USB end) wall of the
front cup, centred at z = 8.00, plus the 15.0 × 9.5 USB-C port (~170 mm²
combined open area onto the board cavity).

## Openings

USB-C port (bottom wall) only — see Rev C/D. The microSD is enclosed (card
installs before the board slides in; +X card channel gives it passage). USER
and PWR buttons and JTAG remain enclosed. The lens is reachable: the barrel
is proud and the Ø16.20 lock ring sits just inside the Ø16.00 aperture, so
focus can be adjusted with the case shut.

## Validated

- 0 interferences between any case part and any of the 17 N6 board solids.
- 0 interference with the battery, seated-SD-card, seated-USB-plug, and
  battery-cable-loop mocks; SD card slides fully out without touching.
- Front cup ∩ back cup = 15.44 mm³, entirely the designed crush-rib
  interference; front cup ∩ plate and back cup ∩ plate both 0.
- All three parts are single valid solids.

## Known compromises

- The front cup ceiling is set by the M12 lens holder (20.25) and lock ring
  (22.85), so there is ~13 mm of dead air above the board at the USB end.
  A stepped front face would take roughly 10 mm off the case but breaks the
  plain slab shape.
- Board is held by 2 screws at the lens end only. Two pads on the plate sit
  0.30 below the PCB at the USB end as anti-bow stops — they do not preload
  the board, and they are placed clear of the header tails and the USB-C
  footprint, but they have not been checked against unmodelled bottom-side
  passives.
- Battery is assumed 30 × 40 × 6. Re-measure and change `BAT_T` in
  `caselib.py` if the real pack is thicker — the case height follows it 1:1.
- PCB thickness 1.30 and header-tail length 3.00 come from the vendor 3D
  model / assumption, not a caliper. Both feed the plate standoff height.
