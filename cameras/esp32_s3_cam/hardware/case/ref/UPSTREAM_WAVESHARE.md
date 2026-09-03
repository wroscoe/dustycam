# Waveshare ESP32-S3-CAM-GC0308 tripod case (body + lid)

Two-piece printed enclosure for the Waveshare ESP32-S3-CAM-GC0308 (sargineer part
`waveshare-esp32s3-cam-gc0308`, 37 x 37 board, 32.6 x 32.6 M2 hole pattern). Files:

- `wsc_cam_case_body.step` / `.step.py` - 44 x 44 x 17 tray: 2 mm walls and floor, 5 mm
  under-board clearance, 4 x dia 5 corner bosses (1.7 mm M2 self-tap pilots, 6 deep) tied to
  the walls by gussets, 13 x 7 USB-C cutout in the Y=0 wall (plug-overmold sized), 8 x 3.5
  lead notch in the +Y wall at floor level (battery / speaker GH1.25 leads), and a dia 14 x
  13 boss on the -X wall for a 1/4"-20 heat-set insert (8.0 hole, ruthex RX-1/4-20; a
  tapered E-Z LOK needs 9.1 - `TRIPOD_HOLE_D`).
- `wsc_cam_case_lid.step` / `.step.py` - 2 mm lid with a 1.5 mm locating lip (0.2 gap),
  4 x dia 5 posts that land on the PCB around its holes (2.3 through, 4.2 x 1.5 pan-head
  recess on top), and an 18 x 18 camera window centred on the GC0308 head parked on the TF
  slot (head top sits 2.5 mm below the lid underside; 1 mm chamfer on the window for FOV).
- `wsc_cam_case_assembly.step` / `.step.py` - body + lid + sarg's board envelope
  (`amz-esp32s3-cam-gc0308.py`), all in the board's frame.
- `cam_case_common.py` - every parameter; `verify.py` - interference / fit checks;
  `stl/*.stl` - ready to slice.

Frame: the board's frame (origin PCB bottom-left, Z=0 PCB bottom, lens on +Z, USB-C on
the Y=0 edge). Outer 44 x 44 x 19 mm (57 over the tripod boss); case edge clearance 1.5 mm
around the PCB, corner R3.

Assembly: drop the board into the body on its bosses (USB-C into the Y=0 cutout), seat the
camera head on the TF slot under the lid window, fit the lid, 4 x M2 x 16 self-tapping pan
head screws from the top: lid -> post -> PCB hole -> boss. One screw set holds both lid and
board. TF card: lift the lid. On a tripod the -X wall is "down", the lens looks horizontal
and USB-C exits sideways.

Print: body floor-down (USB cutout bridges 13 mm, tripod boss prints as a horizontal
cylinder - use supports under it or print the boss with a small flat), lid top-down
(posts straight up, no supports). PETG or PLA, >= 3 perimeters for the bosses.

Status: designed from sarg's envelope model of the board, NOT yet test-fitted. The board
envelope's heights, hole dia (2.5) and USB-C centre are estimates - caliper before trusting
the fit. Not in this case: BOOT/RESET button access and mic ports (positions not in the
part facts) - add holes to `lid()` once measured. `verify.py` reports 8.4 mm3 overlap
between two bosses and the envelope's coarse 30 x 24 underside "connector field" block; the
real connectors keep clear of the mounting holes.
