"""n6_speedcam carrier board (../../carrier/) — envelope for the case checks.

Local frame: origin at the PCB's bottom-left corner on the PCB BOTTOM face,
+Z toward the component side. Geometry comes straight from
index.circuit.tsx: 16 x 12 grid units of 2.54, mounting holes at
(+-7.5U, +-4.5U) about the centre, three headers whose Dupont plugs are the
tallest thing on the board.

Modelled:
  * PCB 40.64 x 30.48 x 1.6;
  * a 8.5 mm part slab over the whole board (male headers, TO-92, the
    100 uF cans lie within that), corners cleared for the screws;
  * Dupont plug envelopes on JR (8 pins), JN (6) and JP (4): 2.54 per pin,
    14 mm tall above the PCB. Use right-angle headers and these fold flat.
"""
from build123d import *

U = 2.54
W, L = 16 * U, 12 * U
PCB_T = 1.6
PART_H = 8.5
PLUG_H = 14.0
HOLE_D = 2.7
CORNER_CLEAR = 7.0
CX, CY = W / 2, L / 2
HOLES = [(CX + sx * 7.5 * U, CY + sy * 4.5 * U) for sx in (-1, 1) for sy in (-1, 1)]
# (centre x, centre y, n pins, horizontal?) from the tscircuit layout
HEADERS = [(CX + 0 * U, CY + 4.5 * U, 8, True),     # JR radar
           (CX - 2 * U, CY - 4.5 * U, 6, True),     # JN N6
           (CX - 6.5 * U, CY, 4, False)]            # JP power
TOP = PCB_T + PLUG_H              # 15.6 above the PCB bottom

MIN3 = (Align.MIN, Align.MIN, Align.MIN)
CCMIN = (Align.CENTER, Align.CENTER, Align.MIN)


def _box(x0, y0, z0, dx, dy, dz):
    return Pos(x0, y0, z0) * Box(dx, dy, dz, align=MIN3)


def gen_step():
    pcb = _box(0, 0, 0, W, L, PCB_T)
    for cx, cy in HOLES:
        pcb -= Pos(cx, cy, -1) * Cylinder(HOLE_D / 2, PCB_T + 2, align=CCMIN)
    pcb.label = "carrier_pcb"

    parts = _box(1.0, 1.0, PCB_T, W - 2, L - 2, PART_H)
    for cx, cy in HOLES:
        parts -= Pos(cx, cy, PCB_T - 1) * Cylinder(CORNER_CLEAR / 2, PART_H + 2, align=CCMIN)
    parts.label = "carrier_parts"

    plugs = None
    for cx, cy, n, horiz in HEADERS:
        dx, dy = (n * U, U) if horiz else (U, n * U)
        b = _box(cx - dx / 2, cy - dy / 2, PCB_T, dx, dy, PLUG_H)
        plugs = b if plugs is None else plugs + b
    plugs.label = "carrier_dupont_plugs"

    asm = Compound(children=[pcb, parts, plugs])
    asm.label = "n6_speedcam_carrier"
    return asm


if __name__ == "__main__":
    print(gen_step().bounding_box())
