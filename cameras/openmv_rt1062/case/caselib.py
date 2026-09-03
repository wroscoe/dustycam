"""OpenMV Cam RT1062 (R6) — drip-resistant two-part printed case.

Parts
-----
cup     Front/side/roof shell, open at the bottom (cables) and at the back.
        Lens hole + lens hood on the front face, flat bridged roof with two
        "chimney" pin holes over the edge switches, four corner screw blocks.
        Prints standing on its open bottom, no supports.
plate   Flat back plate.  Carries the board on two M2 posts (M2 x 12 screws
        replace the camera-module screws), two corner rest posts, a 1/4-20
        heat-set-insert boss pointing INTO the case, a U-shaped locating rim,
        and four counterbored M2 screw holes into the cup's blocks.  Prints
        flat, outer face on the bed, no supports.

Frame — identical to DIMENSIONS.md: origin at the PCB bottom-left corner on
the PCB TOP face, +X across (35.56), +Y toward the lens end (44.45, "up" in
use), +Z along the optical axis.  In use the lens looks horizontally, the
Y=0 edge (USB-C + battery) points down, rain comes from +Y.

Everything is built in this frame; the *_print() helpers rotate for the bed.
"""

from build123d import *  # noqa: F403

# ---------------------------------------------------------------------------
# Board envelope (DIMENSIONS.md)
# ---------------------------------------------------------------------------
PCB_X, PCB_Y, PCB_T = 35.56, 44.45, 1.20
LENS_AXIS = (17.78, 36.25)
LENS_BARREL_D = 14.00
LENS_TIP_Z = 29.95
LENS_GLASS_D = 9.70
MOUNT_TOP_Z = 18.95              # M12 holder top
MOUNT_HOLES = [(2.54, 36.14), (32.91, 36.16)]   # Ø3.0, M2 SMT spacer on top
SPACER_D, SPACER_H = 4.35, 3.01
SWITCH_X = [(6.16, 10.76), (24.90, 29.50)]      # side-actuated, +Y edge
SWITCH_Y = (41.57, 45.11)
SWITCH_Z = (0.0, 1.44)
USB_X, USB_Y, USB_Z = (18.99, 28.57), (0.30, 8.01), (-0.84, 3.32)
JST_X, JST_Y, JST_Z = (7.10, 13.10), (0.23, 7.93), (-3.39, 4.96)
SD_X, SD_Y, SD_Z = (23.72, 35.12), (21.17, 33.12), (-2.66, -1.21)
CAMPCB_Y, CAMPCB_Z = (28.0, 44.51), (3.05, 4.65)
HOLDER_X, HOLDER_Y = (5.88, 29.68), (27.75, 44.75)
BOTTOM_MIN_Z = -3.39

# Plug/cable envelopes hanging off the Y=0 edge (kept clear by the design)
USB_PLUG_X = (17.3, 30.3)        # 13 mm overmold centred on the receptacle
USB_PLUG_Z = (-2.8, 5.2)
JST_PLUG_X = (6.6, 13.6)

# ---------------------------------------------------------------------------
# Case parameters
# ---------------------------------------------------------------------------
CLR = 0.30                  # board edge -> cavity wall, per side
WALL = 3.00                 # side walls
FRONT_T = 2.40              # front wall
ROOF_T = 2.40
PLATE_T = 2.40
ROOF_CHAMFER = 2.00         # cosmetic drip chamfer on outer top edges

Y_BOT = -6.00               # open bottom edge (walls stop here, cables exit)
Y_CEIL = 50.00              # cavity ceiling (switch tips at 45.11 + block room)
Y_TOP = Y_CEIL + ROOF_T     # 52.4

Z_PLATE_IN = -14.00         # plate inner face (battery bay behind the board)
Z_PLATE_OUT = Z_PLATE_IN - PLATE_T          # -16.4
ROOF_LIP = 2.00             # roof continues past the plate as a drip lip
Z_ROOF_BACK = Z_PLATE_OUT - ROOF_LIP        # -18.4
Z_FRONT_IN = MOUNT_TOP_Z + 0.55             # 19.5, just above the M12 holder
Z_FRONT_OUT = Z_FRONT_IN + FRONT_T          # 21.9; lens stands 8.05 proud

CAV_X0, CAV_X1 = -CLR, PCB_X + CLR          # -0.3 .. 35.86
OUT_X0, OUT_X1 = CAV_X0 - WALL, CAV_X1 + WALL   # -3.3 .. 38.86

LENS_HOLE_D = 16.00         # knurled focus ring measured ~Ø15.5 on the real lens (Ø15 bound); 0.25 radial clearance on the teeth

# Lens hood: tube with a flat (bridged) ceiling, cut back 45° underneath so
# it prints standing and the bottom half of the tip is open (drainage, FOV).
HOOD_L = 12.00              # from the front face; tip 3.95 past the glass
HOOD_R_IN = 10.50
HOOD_T = 2.00
HOOD_R_OUT = HOOD_R_IN + HOOD_T
HOOD_FLAT_IN = 8.50         # flat ceiling this far above the axis (12.3 bridge)
HOOD_FLAT_OUT = HOOD_FLAT_IN + HOOD_T

# Corner screw blocks (cup) / plate screws: 4 x M2 self-tapping
BLOCK_W = 5.00              # x extent of each block from the cavity wall
BLOCK_BOT_Y = (Y_BOT, -0.50)            # below the board's bottom edge
BLOCK_TOP_Y = (45.50, Y_CEIL)           # above the switch tips
BLOCK_SETBACK = 1.60        # blocks start this far in front of the plate (rim room)
BLOCK_Z0 = Z_PLATE_IN + BLOCK_SETBACK
M2_PILOT_D = 1.70
M2_PILOT_DEPTH = 8.00
M2_CLEAR_D = 2.40
M2_CBORE_D, M2_CBORE_H = 4.40, 1.20
BLOCK_X = [(CAV_X0, CAV_X0 + BLOCK_W), (CAV_X1 - BLOCK_W, CAV_X1)]
SCREW_XY = [
    ((CAV_X0 + CAV_X0 + BLOCK_W) / 2, sum(BLOCK_BOT_Y) / 2),
    ((CAV_X1 - BLOCK_W + CAV_X1) / 2, sum(BLOCK_BOT_Y) / 2),
    ((CAV_X0 + CAV_X0 + BLOCK_W) / 2, sum(BLOCK_TOP_Y) / 2),
    ((CAV_X1 - BLOCK_W + CAV_X1) / 2, sum(BLOCK_TOP_Y) / 2),
]

# Switch access: pin holes through the roof on short chimneys
CHIMNEY_D, CHIMNEY_H, PIN_HOLE_D = 7.00, 3.00, 2.60
SWITCH_PIN_XZ = [((a + b) / 2, (SWITCH_Z[0] + SWITCH_Z[1]) / 2) for a, b in SWITCH_X]

# microSD card relief in the right wall (card ejects toward +X)
SD_POCKET_DEPTH = 1.50
SD_POCKET_Y = (SD_Y[0] - 1.7, SD_Y[1] + 1.4)
SD_POCKET_Z = (Z_PLATE_IN - 1.0, -0.80)   # full-length groove: card is in during slide-in

# Plate features
RIM_T, RIM_H, RIM_CLR = 1.50, 1.50, 0.20
BOARD_POST_D = 5.00         # M2 posts under the mounting holes
BOARD_POST_TOP_Z = -1.40    # bears on the spacer flange under the PCB
REST_POST_D = 3.40          # bottom-corner rests under the PCB (unpopulated header pads)
REST_POST_XY = [(1.40, 1.60), (PCB_X - 1.40, 1.60)]
REST_POST_TOP_Z = -1.50
INSERT_BOSS_D = 15.00       # 1/4-20 heat-set insert boss (points into the case)
INSERT_BOSS_XY = (LENS_AXIS[0], 42.00)
INSERT_BOSS_TOP_Z = -3.00   # 0.55 below the caps at -2.45
INSERT_HOLE_D = 8.80        # for a 9.5 mm OD x 12.7 long brass 1/4-20 insert (adjust to datasheet)
INSERT_HOLE_DEPTH = 13.00   # from the outer face; 0.4 floor left

PLATE_TOP_Y = Y_CEIL - RIM_CLR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box from min/max corners."""
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def zcyl(x, y, z0, z1, d):
    """Cylinder along +Z from z0 to z1 at (x, y)."""
    return Pos(x, y, (z0 + z1) / 2) * Cylinder(d / 2, z1 - z0)


def ycyl(x, z, y0, y1, d):
    """Cylinder along +Y from y0 to y1 at (x, z)."""
    return Pos(x, (y0 + y1) / 2, z) * Rot(90, 0, 0) * Cylinder(d / 2, y1 - y0)


def _hood_profile(r, flat):
    """Circle of radius r clipped by a flat at +flat above the centre (XY sketch)."""
    ax, ay = LENS_AXIS
    circ = Pos(ax, ay) * Circle(r)
    clip = Pos(ax, ay + flat + 50) * Rectangle(4 * r, 100)
    return circ - clip


def hood():
    outer = extrude(_hood_profile(HOOD_R_OUT, HOOD_FLAT_OUT), HOOD_L)
    inner = extrude(_hood_profile(HOOD_R_IN, HOOD_FLAT_IN), HOOD_L + 2)
    tube = Pos(0, 0, Z_FRONT_OUT) * outer - Pos(0, 0, Z_FRONT_OUT - 1) * inner
    # 45° cut-back: keep y >= (axis - R_out) + (z - Z_FRONT_OUT)
    ax, ay = LENS_AXIS
    y0 = ay - HOOD_R_OUT
    z0, z1 = Z_FRONT_OUT - 0.01, Z_FRONT_OUT + HOOD_L + 1
    # polygon in the YZ plane (local x = global Y, local y = global Z)
    pts = [(y0, z0), (y0 + (z1 - z0), z1), (y0 - 60, z1), (y0 - 60, z0)]
    wedge = extrude(Plane.YZ.offset(ax - HOOD_R_OUT - 1) * Polygon(*pts, align=None),
                    2 * HOOD_R_OUT + 2)
    return tube - wedge


def cup():
    ax, ay = LENS_AXIS
    body = box(OUT_X0, OUT_X1, Y_BOT, Y_TOP, Z_PLATE_IN, Z_FRONT_OUT)
    body += box(OUT_X0, OUT_X1, Y_CEIL, Y_TOP, Z_ROOF_BACK, Z_PLATE_IN)     # roof lip
    top_edges = body.faces().sort_by(Axis.Y)[-1].edges()
    body = chamfer(top_edges, ROOF_CHAMFER)
    # cavity: open at the bottom (Y_BOT) and the back (Z_PLATE_IN)
    body -= box(CAV_X0, CAV_X1, Y_BOT - 1, Y_CEIL, Z_PLATE_IN - 5, Z_FRONT_IN)
    # corner screw blocks
    for (bx0, bx1) in BLOCK_X:
        for (by0, by1) in (BLOCK_BOT_Y, BLOCK_TOP_Y):
            body += box(bx0, bx1, by0, by1, BLOCK_Z0, Z_FRONT_IN)
    for (sx, sy) in SCREW_XY:
        body -= zcyl(sx, sy, BLOCK_Z0 - 1, BLOCK_Z0 + M2_PILOT_DEPTH, M2_PILOT_D)
    # lens hole + hood
    body -= zcyl(ax, ay, Z_FRONT_IN - 1, Z_FRONT_OUT + 1, LENS_HOLE_D)
    body += hood()
    # switch pin chimneys
    for (cx, cz) in SWITCH_PIN_XZ:
        body += ycyl(cx, cz, Y_TOP - 0.5, Y_TOP + CHIMNEY_H, CHIMNEY_D)
        body -= ycyl(cx, cz, Y_CEIL - 1, Y_TOP + CHIMNEY_H + 1, PIN_HOLE_D)
    # microSD relief in the right wall
    body -= box(CAV_X1, CAV_X1 + SD_POCKET_DEPTH, *SD_POCKET_Y, *SD_POCKET_Z)
    body.label = "rt1062_cup"
    return body


def plate():
    p = box(OUT_X0, OUT_X1, Y_BOT, PLATE_TOP_Y, Z_PLATE_OUT, Z_PLATE_IN)
    # U-shaped locating rim entering the cavity
    rx0, rx1 = CAV_X0 + RIM_CLR, CAV_X1 - RIM_CLR
    rz0, rz1 = Z_PLATE_IN, Z_PLATE_IN + RIM_H
    p += box(rx0, rx0 + RIM_T, Y_BOT, PLATE_TOP_Y, rz0, rz1)
    p += box(rx1 - RIM_T, rx1, Y_BOT, PLATE_TOP_Y, rz0, rz1)
    p += box(rx0, rx1, PLATE_TOP_Y - RIM_T, PLATE_TOP_Y, rz0, rz1)
    # board posts (M2 pilot) and corner rests
    for (hx, hy) in MOUNT_HOLES:
        p += zcyl(hx, hy, Z_PLATE_IN - 0.01, BOARD_POST_TOP_Z, BOARD_POST_D)
        p -= zcyl(hx, hy, BOARD_POST_TOP_Z - M2_PILOT_DEPTH, BOARD_POST_TOP_Z + 1, M2_PILOT_D)
    for (rx, ry) in REST_POST_XY:
        p += zcyl(rx, ry, Z_PLATE_IN - 0.01, REST_POST_TOP_Z, REST_POST_D)
    # 1/4-20 insert boss, hole from the outer face
    bx, by = INSERT_BOSS_XY
    p += zcyl(bx, by, Z_PLATE_IN - 0.01, INSERT_BOSS_TOP_Z, INSERT_BOSS_D)
    p -= zcyl(bx, by, Z_PLATE_OUT - 1, Z_PLATE_OUT + INSERT_HOLE_DEPTH, INSERT_HOLE_D)
    # four M2 screw holes with counterbores on the outer face
    for (sx, sy) in SCREW_XY:
        p -= zcyl(sx, sy, Z_PLATE_OUT - 1, Z_PLATE_IN + RIM_H + 1, M2_CLEAR_D)
        p -= zcyl(sx, sy, Z_PLATE_OUT - 1, Z_PLATE_OUT + M2_CBORE_H, M2_CBORE_D)
    p.label = "rt1062_plate"
    return p


def board_mock():
    """Simplified R6 envelope for interference checks (same frame)."""
    ax, ay = LENS_AXIS
    m = box(0, PCB_X, 0, PCB_Y, -PCB_T, 0)
    m += box(0, PCB_X, *CAMPCB_Y, *CAMPCB_Z)
    m += box(*HOLDER_X, *HOLDER_Y, CAMPCB_Z[1], MOUNT_TOP_Z)
    m += zcyl(ax, ay, 13.25, LENS_TIP_Z, LENS_BARREL_D)
    m += box(*USB_X, *USB_Y, *USB_Z)
    m += box(*JST_X, *JST_Y, *JST_Z)
    m += box(*SD_X, *SD_Y, *SD_Z)
    m += box(SD_X[1], SD_X[1] + 1.4, SD_Y[0] + 1, SD_Y[1] - 1, -2.5, -1.4)   # card proud
    m += box(23.98, 29.06, 14.65, 21.0, 0.42, 6.13)                          # SWD header
    caps = box(1.0, 34.5, 1.0, 44.0, -2.45, -PCB_T)                          # caps under
    for (hx, hy) in MOUNT_HOLES:                                             # (post footprints are flange/pad)
        caps -= zcyl(hx, hy, -3, 0, BOARD_POST_D + 0.2)
    for (rx, ry) in REST_POST_XY:
        caps -= zcyl(rx, ry, -3, 0, REST_POST_D + 0.2)
    m += caps
    for (sx0, sx1) in SWITCH_X:
        m += box(sx0, sx1, *SWITCH_Y, *SWITCH_Z)
    for (hx, hy) in MOUNT_HOLES:
        m += zcyl(hx, hy, 0, SPACER_H, SPACER_D)
        m += zcyl(hx, hy, 4.28, 5.95, 4.0)
    m.label = "rt1062_board_mock"
    return m


def usb_plug_mock():
    """USB-C plug overmold hanging out of the bottom (must stay clear)."""
    return box(*USB_PLUG_X, Y_BOT - 10, USB_Y[1], *USB_PLUG_Z)


def cup_print():
    """Cup standing on its open bottom: +Y -> +Z, bottom edge on Z=0."""
    c = cup().rotate(Axis.X, 90)
    bb = c.bounding_box()
    return c.move(Location((-bb.min.X, -bb.min.Y, -bb.min.Z)))


def plate_print():
    """Plate outer face on the bed (features already point +Z)."""
    p = plate()
    bb = p.bounding_box()
    return p.move(Location((-bb.min.X, -bb.min.Y, -bb.min.Z)))
