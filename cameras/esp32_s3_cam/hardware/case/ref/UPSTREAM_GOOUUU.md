# GOOUUU ESP32-S3-CAM tripod case (body + lid)

Two-part printed enclosure for the GOOUUU ESP32-S3-CAM (sargineer `goouuu-esp32s3cam`): a
40-pin DevKitC-style ESP32-S3-WROOM-1 board with an OV3660 on a short flex, two USB-C on one end,
BOOT/RST buttons, a WS2812 and a microSD slot underneath.

* `goouuu_cam_case_body.step` / `.stl` - tub, 79 x 37 x 21.5 (+12.5 tripod boss on the -Y wall)
* `goouuu_cam_case_lid.step` / `.stl` - 2 mm lid with lens window, button + LED holes, corner feet, can pad
* `goouuu_cam_case_assembly.step` - body + lid + board envelope in the board's frame
* `goouuu_esp32s3cam_board.step` - the envelope the case was built against (`goouuu_board_ref.py`)
* `goouuu_cam_case_common.py` - every parameter; `verify.py` - interference / probe checks

## Frame and the board figures used

Board frame: origin PCB plan bottom-left, +X along the long edge, USB-C on the X=0 end, Z=0 PCB
bottom, lens looks +Z. The warehouse board model was an estimated envelope, so the board was
re-measured off the Keyestudio MB0184 ESP32-S3 CAM drawing/photos (same board) using the 2.54 mm
pin pitch as the scale: PCB 63 x 29 x 1.6; 2x20 pins at x = 11.0 + 2.54k on rows y 1.8 / 27.2
(25.4 apart, plastic underneath, pins down); corner holes dia 2.4 at (2.4, 2.0) / (2.4, 27.0);
USB-C shells y 4.2..13.0 (TTL) and 15.6..24.4 (OTG), 1.3 past the edge; BOOT/RST ~(14.25, 6.35) /
(14.25, 22.65); LED (24, 5.7); FPC x 31..40; WROOM x 42.5..68 (antenna past the PCB end); lens
(46.2, 15.0), top 8.5 above the PCB; microSD x 44..60 underneath. Treat all of it as +/-0.5 mm.

## How the board is held

No screws go through the board. The two header rows drop into 1.5 mm grooves in two bed rails
(25.4 apart - the one dimension that is certain); the header plastic seats on the rails so the
PCB bottom lands at Z=0. The body's USB-end corner blocks stop the PCB edge and carry ledges
under the corners; the lid's corner feet clamp the corners (tapered pins probe the corner holes
- snip them if they miss) and a pad hovers 0.3 mm over the WROOM can at the other end. Four
M2 x 8 self-tappers hold the lid in full-height corner bosses.

## Openings

USB: one 23.8 x 8 opening in the X=0 wall takes both plug overmolds (12 x 6.5), ports recessed
5.2 mm. Lens: dia 14 window, 1 mm chamfer. Buttons: dia 6 holes (press with a pen). LED: dia 3.
microSD: 15 x 4 slot in the +X wall (card mouth ~10 mm inside - tweezers). Tripod: 1/4"-20
heat-set insert boss on the -Y wall (dia 14 x 12.5, hole 8 x 13.5) so the lens looks sideways.

## Printing

Body floor-down (support the side boss), lid top-down; 0.4 nozzle / 0.2 layer, PLA/PETG,
>= 3 perimeters. Hardware: 4 x M2 x 8 pan head, 1 x ruthex RX-1/4-20 (optional).

## Caveats

Unprinted, unfitted. Risk items in order: header pin length below the plastic (6.0 assumed ->
groove depth 6.5); lens height (8.5 assumed -> lid underside Z 11); corner-hole / button / LED
positions (+/-0.5). Caliper the owned unit, update `goouuu_board_ref.py`, re-run `verify.py`.
