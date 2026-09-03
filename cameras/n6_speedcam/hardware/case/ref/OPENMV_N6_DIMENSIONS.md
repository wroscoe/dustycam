# OpenMV Cam N6 — measured dimensions

All values in mm. **Frame:** origin at PCB bottom-left corner, Z = 0 at PCB
**bottom** face, +Y toward the camera end, +Z along the optical axis. Matches
the orientation of the official pinout diagram. `ZT` = 1.30 = PCB top face.

## Provenance

Measured from OpenMV's own 3D models, not estimated:

| Source | Use |
|---|---|
| `github.com/openmv/openmv-boards` → `models/OPENMV_N6.glb` | Draco-compressed; per-primitive bounds gave component envelopes |
| `openmv.io/cdn/shop/3d/models/o/d94e61da64841341/N6_R3_Render.glb` | Uncompressed; PCB mesh vertices gave the exact outline + every drilled hole |
| `github.com/openmv/openmv-datasheets` → `products/cameras/openmv-n6.yaml` | "2 mounting holes", M12 (S-mount) lens |
| `openmv.io/products/openmv-n6` photos + pinout PNG | Visual confirmation of which holes are which |

Vendor product page quotes 45 × 35 × 30 mm / 21 g. The 30 mm is PCB-top to
lens-top; measured here as 29.95.

## PCB

| Feature | Value |
|---|---|
| Outline | 35.56 × 44.45 (= 1.400 × 1.750 in), square corners |
| Thickness | 1.30 |
| Overall envelope | 35.81 × 45.08 × 34.25 (incl. camera-PCB and button overhang) |

## Holes

| Hole | Ø | Positions (x, y) | Notes |
|---|---|---|---|
| Mounting | 2.80 | (3.048, 41.402), (32.512, 41.402) | 29.464 apart. Gold-ringed; the "2 mounting holes" |
| Camera standoff | 3.00 | (2.540, 36.195), (33.020, 36.195) | 30.480 apart. Würth 9774030243R SMT spacers press in |
| Header pins ×32 | 1.02 | x ∈ {1.600, 4.140, 31.460, 34.000}, y = 1.599 + 2.54·n, n = 0…7 | 2.54 pitch |

Plus 5 small board cutouts serving the USB-C shell tabs and the LiPo connector.

## Optics

| Feature | Value |
|---|---|
| Optical axis | (17.810, 36.255) — centred in X, 8.20 from the +Y edge |
| Mount | M12 × 0.5 (S-mount), interchangeable |
| Lens holder | 23.80 × 17.00, z = ZT+4.65 … ZT+18.95 |
| Lens barrel | Ø14.00, z = ZT+13.25 … ZT+29.95 |
| Lock ring | Ø16.20, z = ZT+18.95 … ZT+21.55 |
| Front element | Ø9.70, recessed 0.20 |
| Camera daughter PCB | 1.60 thick, z = ZT+3.048 … ZT+4.648, spans y 28.0 … 44.51 |

## I/O and components — (x0, y0, dx, dy, z0, dz), z relative to ZT

| Part | Spec |
|---|---|
| USB-C receptacle (mid-mount) | 18.88, 0.27, 9.58, 7.53, −0.85, 4.16 |
| microSD socket (**bottom** side) | 23.62, 21.15, 11.40, 11.95, −2.76, 1.45 |
| LiPo battery connector (2-pin) | 7.00, 0.20, 6.00, 7.70, 0.00, 4.96 |
| JTAG/SWD 2×5, 1.27 pitch | 23.89, 14.63, 5.08, 6.35, 0.42, 5.71 |
| BOOT1 switch | 32.75, 21.50, 2.50, 8.00, 0.00, 2.49 |
| Camera board-to-board (DF12) | 11.73, 38.34, 12.10, 4.60, 0.00, 3.048 |
| Button USER / SW | 6.18, 41.53, 4.60, 3.55, 0.00, 1.43 (overhangs +Y by 0.63) |
| Button PWR / SW2 | 24.80, 41.53, 4.60, 3.55, 0.00, 1.43 (overhangs +Y by 0.63) |
| Headers, 2 × 2×8 female | x 0.33…5.41 and 30.19…35.27, y 0.32…20.64, 8.50 tall |

## Not modelled

Silkscreen, ICs, passives, LEDs, the antenna, Ø0.6 locating vias, and the slots
in the camera daughter-board arms (modelled as a plain cross). Header tail
length below the PCB (3.0) is assumed, not measured. PCB thickness 1.30 is the
vendor model's value — worth a caliper check against a physical board.
