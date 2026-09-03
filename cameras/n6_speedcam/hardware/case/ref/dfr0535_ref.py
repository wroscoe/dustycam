"""DFRobot DFR0535 "Solar Power Manager" — envelope model for the case checks.

Local frame: origin at the PCB's bottom-left corner on the PCB BOTTOM face,
+Z toward the component side. The case mounts it component side forward
(+Z), rotated 180 deg in-plane relative to DFRobot's overview photo, so that
BAT IN / SOLAR IN end up on the RIGHT edge (next to the cable gland) and
OUT1-3 on the LEFT edge (next to the carrier).

Known (wiki + datasheet PDF in hardware/power/): 78.0 x 68.0 mm; 4 corner
mounting holes; screw terminals on two opposite edges; USB-A OUT on a third
edge; heatsink pad on the back.  Modelled, NOT measured:
  * PCB 1.6 thick;
  * holes Ø3.2 at 3.5 in from each corner (71 x 61 pattern) -- caliper it;
  * component envelope = one 11 mm slab inset 1.5 from the edges, corners
    cleared for the screws (terminal blocks are ~10 tall, USB-A ~8);
  * the optional kit heatsink as 14 x 14 x 6 on the back, at the edge
    OPPOSITE the USB-A socket (top centre in this orientation).
"""
from build123d import *

W, L = 78.0, 68.0
PCB_T = 1.6
COMP_H = 11.0
COMP_INSET = 1.5
HOLE_D = 3.2
HOLE_IN = 3.5
HOLES = [(HOLE_IN, HOLE_IN), (W - HOLE_IN, HOLE_IN),
         (HOLE_IN, L - HOLE_IN), (W - HOLE_IN, L - HOLE_IN)]
CORNER_CLEAR = 7.5
HEATSINK = dict(x0=W / 2 - 7.0, y0=L - 4.0 - 14.0, dx=14.0, dy=14.0, dz=6.0)
TOP = PCB_T + COMP_H              # 12.6 above the PCB bottom
BACK = HEATSINK["dz"]             # 6.0 behind it

MIN3 = (Align.MIN, Align.MIN, Align.MIN)
CCMIN = (Align.CENTER, Align.CENTER, Align.MIN)


def _box(x0, y0, z0, dx, dy, dz):
    return Pos(x0, y0, z0) * Box(dx, dy, dz, align=MIN3)


def gen_step():
    pcb = _box(0, 0, 0, W, L, PCB_T)
    pcb = fillet(pcb.edges().filter_by(Axis.Z), 3.0)
    for cx, cy in HOLES:
        pcb -= Pos(cx, cy, -1) * Cylinder(HOLE_D / 2, PCB_T + 2, align=CCMIN)
    pcb.label = "dfr0535_pcb"

    comp = _box(COMP_INSET, COMP_INSET, PCB_T, W - 2 * COMP_INSET, L - 2 * COMP_INSET, COMP_H)
    for cx, cy in HOLES:
        comp -= Pos(cx, cy, PCB_T - 1) * Cylinder(CORNER_CLEAR / 2, COMP_H + 2, align=CCMIN)
    comp.label = "dfr0535_components"

    hs = _box(HEATSINK["x0"], HEATSINK["y0"], -HEATSINK["dz"], HEATSINK["dx"], HEATSINK["dy"], HEATSINK["dz"])
    hs.label = "dfr0535_heatsink_back"

    asm = Compound(children=[pcb, comp, hs])
    asm.label = "dfr0535_solar_power_manager"
    return asm


if __name__ == "__main__":
    print(gen_step().bounding_box())
