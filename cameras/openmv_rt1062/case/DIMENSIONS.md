# OpenMV Cam RT1062 (R6 board) — measured dimensions

All values in mm. **Frame:** origin at the PCB bottom-left corner on the PCB
**top** face (component/lens side). +X across the board (35.56), +Y toward the
lens end (44.45), +Z along the optical axis. PCB occupies Z −1.20 … 0.

## Provenance

Measured with trimesh from OpenMV's own Shopify 3D viewer models (uncompressed
GLB, metres):

| File (in `ref/`) | Source | Use |
|---|---|---|
| `RT1062_R6.glb` | `openmv.io/cdn/shop/3d/models/o/4ee89ee9a4fba5f9/RT1062_R6.glb` | R6 board: outline, holes, every component envelope |
| `Case-rt1062-V4.glb` | `openmv.io/cdn/shop/3d/models/o/3b6ad28ca0cdb0eb/Case-rt1062-V4.glb` | official V4 case (existing design, for reference) |
| `openmv-cam-rt.yaml` | `github.com/openmv/openmv-datasheets` | "2 mounting holes", M12 lens, 70 g |

Also on the CDN, not copied: `RT1062_R5.glb`, `ov5640.glb`, `cam-v5.glb`.
Vendor page quotes 45 × 36 × 29 mm; measured 44.45 × 35.56 × (29.95 + 1.2).

## Board

| Item | Value |
|---|---|
| PCB outline | 35.56 × 44.45, square corners, 1.20 thick (Z −1.20 … 0) |
| Mounting holes | Ø3.0 at (2.54, 36.14) and (32.91, 36.16) — 30.37 apart |
| On those holes | Würth 9774030243R SMT steel spacer, **M2 internal thread**, 3.0 tall, Ø4.35, on the TOP face (Z 0 … 3.01), flange trace to Z −1.39 |
| Camera module screws | 2 × M2 pan head (heads Ø4, Z 4.28 … 5.95) into the spacers |
| Camera module PCB | full width, Y 28.0 … 44.51, Z 3.05 … 4.65 |
| Lens mount (holder) | X 5.88 … 29.68, Y 27.75 … 44.75, Z 4.65 … 18.95 |
| Lens axis | (17.78, 36.25) |
| Lens barrel | Ø14.0 from Z ≈ 15 to the tip at Z 29.95 (M12 thread Ø12 below Z 15); front element Ø9.7 |
| Side-actuated tact switches | KMS231GLFS ×2 on the +Y edge: X 6.16 … 10.76 and 24.90 … 29.50, Y 41.57 … 45.11 (0.66 proud of the edge), Z 0 … 1.44; actuated by pushing in −Y |
| USB-C (USB4105) | X 18.99 … 28.57, Y 0.30 … 8.01, Z −0.84 … 3.32, opening on the Y = 0 edge |
| Battery connector (JST PH-2 style) | X 7.10 … 13.10, Y 0.23 … 7.93, Z −3.39 … 4.96, opening on the Y = 0 edge |
| microSD | X 23.72 … 35.12, Y 21.17 … 33.12, Z −2.66 … −1.21 (bottom side); card ejects toward +X |
| SWD header (FTSH-105) | X 23.98 … 29.06, Y 14.65 … 21.0, Z 0.42 … 6.13 |
| Tallest top-side part outside the lens zone | 6.13 (SWD header) |
| Lowest bottom-side part | −3.39 (battery connector); caps −2.45; SD −2.66 |
| 2×8 header footprints | both long edges, Y 0.35 … 20.7; ship unpopulated (the model shows them populated: Z −10.49 … 8.5 if fitted) |

## Case-relevant conclusions

- The board can be fastened only through the two M2 spacers: replace the two
  camera-module screws with **M2 × 12** that pass camera PCB → spacer → PCB
  → into a post behind the board (the official V4 case does the same with
  hex standoffs).
- Anything that must clear the board from behind needs Z ≤ −3.4 at the
  battery connector and Z ≤ −2.75 under the caps near the top edge.
- The two edge switches need access from +Y (a pin pushed down through the
  roof works).
