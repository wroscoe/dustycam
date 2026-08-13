# Enclosure

3D-printable weatherproof case for the DustyCam.

## Layout

- `src/` — parametric CAD source. Preferred format is Python
  (CadQuery/build123d) so the case regenerates from parameters (board variant,
  lens length, mount type). Native files from GUI CAD (`.f3d`, `.FCStd`) are
  acceptable but commit an exported STEP alongside them.
- `export/pi5/`, `export/pi-zero/` — ready-to-print STEP + STL/3MF per board
  variant. Regenerated from `src/`; don't hand-edit.

## Design requirements (draft)

- Fits board + camera + buck converter only — the battery/solar pack is a
  separate shared box (see `../power/README.md`), which keeps this case
  small. Check clearances against real STEP models of the boards, not
  datasheet rectangles.
- Power entry is the 12–24 V bus: one 5.5×2.1 mm barrel jack (or gland) —
  no battery compartment.
- Lens opening sized for C/CS-mount lenses on the HQ camera; keep the front
  face printable without supports.
- Cable/gland entry for power; vent or desiccant pocket for condensation.
- Mounting: pole strap slots and/or 1/4"-20 tripod insert.
- Print orientation and material notes go in a comment at the top of the
  source file (outdoor use → PETG/ASA, not PLA).
