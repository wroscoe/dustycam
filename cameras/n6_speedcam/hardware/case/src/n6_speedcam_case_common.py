"""n6_speedcam case: shared parameters + part builders (body + lid).

One box holding the OpenMV N6, the HLK-LD2415H radar, the carrier board, a
DFRobot DFR0535 solar manager and a 1S LiPo. Units mm.

FRAME = the N6 board frame (ref/openmv_n6_ref.py): origin at the N6 PCB's
bottom-left corner on its bottom face, +X across the N6, +Y toward its lens
end, +Z along the optical axis. Every other part is placed in that frame.

DEPLOYED POSE -- same rule as the ESP32-S3 cases (hardware_common/caseskit.py):
the case stands on the N6's USB edge, so

    board -Y  ->  DOWN     N6 USB-C, the cable gland and the drain open
                           through the floor
    board +Y  ->  UP       the roof carries the rain hood
    board +Z  ->  FORWARD  the lid is the front: lens window + radome

Split of duties:
  LID  (front)  carries every sensor: the radar screwed to four short bosses
                inside a thinned radome pocket; the N6 hung from two long
                bosses through its Ø2.80 mounting holes; the carrier on four
                short bosses above the N6. The lid's outer face is FLAT, so
                it prints outer-face-down with all bosses growing upward.
  BODY (back)   carries the power: the DFR0535 on four posts off the back
                wall, the LiPo pouch taped to the back wall BEHIND the
                DFR0535 (sandwiched, nothing rigid bears on it), the 1/4"-20
                insert column, the rain hood (a plain extension of the roof
                wall, so it prints as more wall), the three-sided frame the
                lid slides into, and the floor openings. Prints
                back-face-down; the back face is flat because the insert
                pocket is blind inside its column.

THE HINGED FRONT. The lid is a battery-door: its top edge tucks into a
pocket under the hood (a lip hanging in front of it, a 45-degree rail behind
it), then it hinges down and its bottom and side lips press into the cavity
mouth on crush ribs. Nothing is fastened on the front face. So:
  * the top seam faces the sky only through the hood and the lip -- rain
    running down the hood drips off its tip 15 mm out from the face;
  * rain on the face runs straight down, meets the lens barrel (which
    stands LENS_PROUD of the face) and parts around it, and leaves off the
    lid's chamfered bottom arris;
  * the side seams are flush lines on the side faces, backed by the side
    lips: a film that wicks in runs down the lip/wall gap and out through
    the underside seam, which faces down and drains by gravity;
  * the bottom seam is on the underside.
Opening: nail into the pry notch on the underside, pull the bottom edge out
(the ribs let go), drop the top edge out of the pocket. Two optional M2.5
screws from the floor into blocks on the lid back make it tamper-resistant
for a roadside install. Residual path: a film across the top rail seat into
the cavity; that lands on the cavity floor and leaves by the drain slot.
Not sealed; rain-resistant.

Weather: caseskit's rules -- every opening faces down, the floor and roof
faces are chamfered, the lens window is a conical flare, the roof hood
throws rain forward of both windows. PETG or ASA, not PLA.

RADOME: 24 GHz through the lid. The lid is pocketed from the inside to
RADOME_T over the radar. Thin (<= ~1.2 mm, well under a tenth of the 7.6 mm
in-material wavelength for PETG er~2.7) or half-wave (~3.8 mm) both
transmit; anything in between reflects a growing share back into the
antenna. 1.2 mm = 6 layers at 0.2 is the thinnest that still prints as a
weather wall. Keep paint, foil labels and metal away from that panel.

PRINT: body back-face-down (flat; the walls and posts grow up, the hood is
just taller roof wall; the pocket rail is a 45-degree wedge; the pocket lip is a 2.5 mm
ledge under the hood and may want a line of support). Lid outer-face-down (flat; the N6 bosses, the pads, the lips
and the two screw blocks grow up). 0.4 nozzle, 0.2 layers, >= 3
perimeters, no supports on either part. The gland hole is a horizontal
Ø12.5 bore in a vertical wall -- PETG bridges it; clean the top with a drill.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent                # .../hardware/case/src
CASE = HERE.parent                                    # .../hardware/case
sys.path[:0] = [str(HERE), str(CASE / "ref"), str(CASE.parents[2] / "hardware_common")]
from pcbkit import *
from build123d import *
import caseskit as K
import openmv_n6_ref as N6
import ld2415h_ref as RD
import dfr0535_ref as PM
import carrier_ref as CR

# --- deployed orientation ----------------------------------------------------------------
DOWN, UP = "-Y", "+Y"

# --- N6 facts used here (ref/openmv_n6_ref.py) ----------------------------------------------
N6_PCB_T = N6.PCB_T                          # 1.30
N6_MOUNT = N6.MOUNT_HOLES                    # (3.048, 41.402), (32.512, 41.402), Ø2.80
CAM_C = N6.LENS_AXIS                         # (17.81, 36.255)
LENS_TIP_Z = N6.ZT + N6.BARREL_Z[1]          # 31.25
USB_XC = N6.USB_C[0] + N6.USB_C[2] / 2       # 23.67
USB_ZC = N6.ZT + N6.USB_C[4] + N6.USB_C[5] / 2   # 2.53
CAM_ARM_TOP_Z = N6.ZT + N6.CAM_PCB_Z[1]      # 5.95: the daughter-board arm under the bosses

# --- case parameters ----------------------------------------------------------------------
WALL = 2.4
BACK_T = 2.4                                 # the -Z wall (deployed BACK)
LID_T = 2.4
CORNER_R = 3.0
CHAMFER_PORT = 1.5                           # floor face arris
CHAMFER_ROOF = 1.5
HOOD_PROJ = 15.0                             # roof hood, forward of the lid face
# lid <-> body: hinge-in door (see module docstring)
LENS_PROUD = 5.0                             # lens tip stands this far out of the lid face
POCKET_GAP = 0.4                             # lid face to lip / lid back to rail, in the top pocket
POCKET_IN = 2.5                              # how far the roof lip hangs down over the lid
POCKET_T = 2.4                               # lip thickness in front of the lid
RAIL_IN = 1.8                                # rail seat width behind the lid's top edge (45 deg)
LID_TOP_CHAMFER = (1.5, 0.8)                 # back, front: lets the top edge rotate in the pocket
LIP_GAP, LIP_T, LIP_H = 0.2, 1.2, 3.0        # side + bottom lips inside the cavity mouth
LIP_Y_TOP = 77.0                             # side lips stop below the pocket rail
RIB_PROUD, RIB_LEN = 0.25, 6.0               # crush ribs: 0.05/side net interference
SIDE_RIB_Y = [5.0, 35.0, 65.0]               # rib starts along each side lip
BOTTOM_RIB_X = [12.0, 45.0, 84.0]            # rib starts along the bottom lip
PRY_NOTCH_W, PRY_NOTCH_D = 12.0, 1.5         # underside, centre of the floor wall's front edge
LID_SCREW_X = [7.0, 73.0]                    # optional 2 x M2.5 x 8 up through the floor
LID_BLOCK = (6.0, 6.0, 7.0)                  # block width (x), height (y), height below Z_CEIL
SCREW_THRU_D, CBORE_D, CBORE_DEPTH = 2.8, 5.0, 1.5
LID_PILOT_D, LID_PILOT_DEPTH = 2.1, 5.5
LID_EDGE_CHAMFER = 1.5                       # drip arris along the lid's bottom-front edge
# radar on the lid
RADAR_X0, RADAR_Y0 = 44.5, 8.0               # radar PCB corner, long axis vertical
RADOME_T = 1.2                               # see module docstring
RADAR_STANDOFF = 5.0                         # boss height off the pocket floor
RADAR_BOSS_D, RADAR_PILOT_D = 6.0, 2.1       # M2.5 x 6
RADOME_MARGIN = 1.0                          # pocket runs this far past the PCB
# N6 on the lid
N6_BOSS_D, N6_TIP_D, N6_TIP_TOP_Z = 5.5, 4.4, 6.5   # tip is narrower to clear the camera arm
N6_PILOT_D, N6_PILOT_DEPTH = 2.1, 6.0        # M2.5 x 8
N6_PADS = [(1.0, 21.5, 4.0, 4.0), (29.5, 21.5, 3.0, 4.0)]   # anti-bow pads (x0, y0, dx, dy)
PAD_GAP = 0.3
# carrier on the lid
CARRIER_C = (21.0, 62.0)                     # PCB centre; component side faces the BACK
CARRIER_STANDOFF = 6.0
CARRIER_BOSS_D, CARRIER_PILOT_D, CARRIER_PILOT_DEPTH = 5.5, 2.1, 4.5   # M2.5 x 6
# DFR0535 in the body
DFR_X0, DFR_Y0 = 5.0, 9.0
DFR_POST_H, DFR_POST_D, DFR_PILOT_D, DFR_PILOT_DEPTH = 16.0, 6.0, 2.5, 8.0   # M3 x 8
# battery: envelope only (x0, y0, dx, dy, dz) against the back wall, behind the DFR0535
BAT = (16.0, 16.0, 60.0, 36.0, 8.0)
# tripod: blind 1/4"-20 pocket inside an internal column, back face stays flat
TRIPOD_XY = (60.0, 62.0)                     # clear of the DFR0535 heatsink pad + battery
TRIPOD_BOSS_D, TRIPOD_HOLE_D, TRIPOD_HOLE_DEPTH = 16.0, 8.0, 13.5   # ruthex RX-1/4-20
TRIPOD_INNER_H = 12.2
# floor openings
USB_CUT_W, USB_CUT_Z0, USB_CUT_Z1 = 14.0, -3.5, 8.0
DRAIN_XC, DRAIN_W, DRAIN_H = 64.0, 8.0, 3.0
GLAND_D, GLAND_XZ = 12.5, (87.0, -15.0)      # PG7 cable gland for the solar lead
# lens window: Ø14 barrel through Ø15, 0.6 flare at the face; lens stands LENS_PROUD out
WINDOW_D, WINDOW_D_OUT = 15.0, 16.2
CASE_COLOR = Color(0.25, 0.45, 0.75)

# --- derived ------------------------------------------------------------------------------
XI0, XI1 = -8.0, 105.0                       # cavity x: corner bosses left, radar boss clear right
YI0, YI1 = -3.0, 80.0                        # cavity y: below the USB shell .. above the radar
Z_CEIL = LENS_TIP_Z - LID_T - LENS_PROUD     # 23.85: lid inner face; lens tip 5 proud of the outer face
Z_TOP = Z_CEIL + LID_T                       # 26.25
Z_POCKET = Z_TOP + POCKET_GAP + POCKET_T     # 30.45: front of the pocket lip under the hood
Z_RAIL = Z_CEIL - POCKET_GAP                 # 23.45: the rail seat behind the lid's top edge
LID_SCREW_Z = Z_CEIL - LID_BLOCK[2] / 2
Z_IN_FLOOR = -33.5                           # back wall inner face; see stack below
Z_BOT = Z_IN_FLOOR - BACK_T                  # -35.9
X0, Y0 = XI0 - WALL, YI0 - WALL
OUT_X, OUT_Y = (XI1 - XI0) + 2 * WALL, (YI1 - YI0) + 2 * WALL
BODY_H = Z_CEIL - Z_BOT                      # walls; the roof strip continues HOOD_PROJ past Z_TOP
LO = (X0, Y0, Z_BOT)
HI_BODY = (X0 + OUT_X, Y0 + OUT_Y, Z_CEIL)
# z stack, back to front (all must nest; verify_n6_speedcam.py checks the volumes):
#   -33.5 back wall | battery to -25.5 | DFR posts to -17.5, PCB, parts to -4.9
#   | N6 tails -3.0, PCB 0..1.3, headers to 9.8 | carrier plugs 7.4..21.4, PCB to 23
#   | radar J4 to 14.2, parts, PCB, antenna face 25.2 | bosses | radome 30.2..31.4
Z_DFR_PCB = Z_IN_FLOOR + DFR_POST_H          # -17.5
Z_RADOME_IN = Z_CEIL + (LID_T - RADOME_T)    # 30.2: pocket floor
Z_RADAR_FACE = Z_RADOME_IN - RADAR_STANDOFF  # 25.2
Z_CARRIER_PCB = Z_CEIL - CARRIER_STANDOFF    # 23.0: PCB top (lid side); parts hang toward -Z
RADAR_HOLES = [(RADAR_X0 + x, RADAR_Y0 + y) for (x, y) in RD.HOLES]
DFR_HOLES = [(DFR_X0 + x, DFR_Y0 + y) for (x, y) in PM.HOLES]
CARRIER_X0, CARRIER_Y0 = CARRIER_C[0] - CR.W / 2, CARRIER_C[1] - CR.L / 2
CARRIER_HOLES = [(CARRIER_X0 + x, CARRIER_Y0 + y) for (x, y) in CR.HOLES]
POCKET = (RADAR_X0 - RADOME_MARGIN, RADAR_Y0 - RADOME_MARGIN,
          RD.W + 2 * RADOME_MARGIN, RD.L + 2 * RADOME_MARGIN)
USB_CUT_XC = USB_XC
LID_Y1 = YI1 - LIP_GAP                       # 79.8: top edge, 0.2 under the roof


def _wedge_along_y(pts_xz, y0, y1):
    """Prism from an (x, z) polygon, running y0..y1."""
    w = extrude(Plane.XZ * Polygon(*pts_xz, align=None), amount=(y1 - y0) / 2, both=True)
    return Pos(0, (y0 + y1) / 2, 0) * w


def _wedge_along_x(pts_yz, x0, x1):
    """Prism from a (y, z) polygon, running x0..x1."""
    w = extrude(Plane.YZ * Polygon(*pts_yz, align=None), amount=(x1 - x0) / 2, both=True)
    return Pos((x0 + x1) / 2, 0, 0) * w


def top_pocket():
    """The lip hanging from the hood in front of the lid's top edge, and the rail behind it."""
    # the lip's underside is FLAT (a sloped one would cam the lid out of the
    # pocket); its lower-front arris gets a small chamfer as a drip edge. In the
    # print it is a POCKET_IN-wide ledge hanging off the hood -- the one place
    # that may want a line of support.
    z0, c = Z_TOP + POCKET_GAP, 0.6
    lip = _wedge_along_x([(YI1 + 0.01, z0), (YI1 - POCKET_IN + c, z0), (YI1 - POCKET_IN, z0 + c),
                          (YI1 - POCKET_IN, Z_POCKET), (YI1 + 0.01, Z_POCKET)], X0, X0 + OUT_X)
    rail = _wedge_along_x([(YI1 + 0.01, Z_RAIL - RAIL_IN), (YI1 - RAIL_IN, Z_RAIL), (YI1 + 0.01, Z_RAIL)], XI0, XI1)
    return lip + rail


def lid_lips():
    """Side and bottom lips inside the cavity mouth (0.2 clear), no lip at the top."""
    g, t, h = LIP_GAP, LIP_T, LIP_H
    left = box((XI0 + g, YI0 + g, Z_CEIL - h), (t, LIP_Y_TOP - YI0 - g, h + 0.01))
    right = box((XI1 - g - t, YI0 + g, Z_CEIL - h), (t, LIP_Y_TOP - YI0 - g, h + 0.01))
    bottom = box((XI0 + g, YI0 + g, Z_CEIL - h), (XI1 - XI0 - 2 * g, t, h + 0.01))
    return left + right + bottom


def crush_ribs():
    """0.25 proud ribs on the lips' outer faces with a 45 deg nose, so the door
    rotates in and the ribs crush 0.05/side against the walls. Designed
    interference -- the verify script expects exactly this volume."""
    g, h, p = LIP_GAP, LIP_H, RIB_PROUD
    z_nose = Z_CEIL - h + 0.01
    ribs = None
    for y in SIDE_RIB_Y:
        l = _wedge_along_y([(XI0 + g + 0.01, Z_CEIL), (XI0 + g - p, Z_CEIL), (XI0 + g - p, z_nose + p),
                            (XI0 + g + 0.01, z_nose)], y, y + RIB_LEN)
        r = _wedge_along_y([(XI1 - g - 0.01, Z_CEIL), (XI1 - g + p, Z_CEIL), (XI1 - g + p, z_nose + p),
                            (XI1 - g - 0.01, z_nose)], y, y + RIB_LEN)
        ribs = l + r if ribs is None else ribs + l + r
    for x in BOTTOM_RIB_X:
        b = _wedge_along_x([(YI0 + g + 0.01, Z_CEIL), (YI0 + g - p, Z_CEIL), (YI0 + g - p, z_nose + p),
                            (YI0 + g + 0.01, z_nose)], x, x + RIB_LEN)
        ribs = ribs + b
    return ribs


def body():
    # One extrusion for walls AND hood: the slab runs all the way to the hood
    # tip, gets its roof/floor chamfers while every face is still a plain
    # rectangle, and only then is everything in front of the lid plane cut
    # away except the roof strip. The hood is therefore the roof wall itself
    # continuing forward -- no union seam, no neck (the first revision hung it
    # off the roof chamfer's 0.9 mm edge; the second left a 22.5-degree groove
    # at the seam, faces o1.1.f59 / o1.1.f35).
    b = slab(OUT_X, OUT_Y, (Z_TOP + HOOD_PROJ) - Z_BOT, r=CORNER_R, at=(X0, Y0, Z_BOT))
    b = K.face_chamfer(b, DOWN, CHAMFER_PORT)
    b = K.face_chamfer(b, UP, CHAMFER_ROOF)         # lands on the hood tip too
    b = b - box((X0 - 1.0, Y0 - 1.0, Z_CEIL), (OUT_X + 2.0, (OUT_Y - WALL) + 1.0, HOOD_PROJ + LID_T + 2.0))
    b = b - box((XI0, YI0, Z_IN_FLOOR), (XI1 - XI0, YI1 - YI0, BODY_H + 20))
    b = b + top_pocket()
    # pry notch: a shallow recess in the floor wall's front edge, under the lid's bottom edge
    xc = (XI0 + XI1) / 2
    b = b - box((xc - PRY_NOTCH_W / 2, Y0 - 1.0, Z_CEIL - PRY_NOTCH_D), (PRY_NOTCH_W, (YI0 - Y0) + 1.0 - 0.6, PRY_NOTCH_D + 1.0))
    # optional lid screws come up through the floor into the lid's blocks
    for x in LID_SCREW_X:
        b = b - (Pos(x, Y0 - 1.0, LID_SCREW_Z) * Rot(-90, 0, 0) * Cylinder(SCREW_THRU_D / 2, WALL + 2.0,
                                                                         align=(Align.CENTER, Align.CENTER, Align.MIN)))
        b = b - (Pos(x, Y0 - 1.0, LID_SCREW_Z) * Rot(-90, 0, 0) * Cylinder(CBORE_D / 2, CBORE_DEPTH + 1.0,
                                                                         align=(Align.CENTER, Align.CENTER, Align.MIN)))
    # DFR0535 posts off the back wall
    for (x, y) in DFR_HOLES:
        b = b + cyl((x, y), Z_IN_FLOOR, DFR_POST_D, DFR_POST_H)
        b = b - cyl((x, y), Z_IN_FLOOR + DFR_POST_H - DFR_PILOT_DEPTH, DFR_PILOT_D, DFR_PILOT_DEPTH + 1)
    # tripod insert: blind pocket inside an internal column; proud=0 keeps the back flat
    b = K.tripod_boss_z(b, TRIPOD_XY, Z_BOT, Z_IN_FLOOR, proud=0.0, inner_h=TRIPOD_INNER_H,
                        boss_d=TRIPOD_BOSS_D, hole_d=TRIPOD_HOLE_D, hole_depth=TRIPOD_HOLE_DEPTH)
    # --- the floor, in the deployed sense: the Y=YI0 wall ---
    b = K.port_slot(b, LO, HI_BODY, DOWN, center_a=USB_CUT_XC, w_a=USB_CUT_W,
                    center_b=(USB_CUT_Z0 + USB_CUT_Z1) / 2, w_b=USB_CUT_Z1 - USB_CUT_Z0, wall_t=WALL)
    # drain at the cavity floor line (z is horizontal once stood up, so the drain
    # sits at the lowest y, which is this wall, and runs to the back wall's inner face)
    b = K.port_slot(b, LO, HI_BODY, DOWN, center_a=DRAIN_XC, w_a=DRAIN_W,
                    center_b=Z_IN_FLOOR + DRAIN_H / 2, w_b=DRAIN_H, wall_t=WALL)
    # PG7 gland bore for the solar lead
    gx, gz = GLAND_XZ
    b = b - (Pos(gx, Y0 - 1.0, gz) * Rot(-90, 0, 0) * Cylinder(GLAND_D / 2, WALL + 2.0,
                                                               align=(Align.CENTER, Align.CENTER, Align.MIN)))
    b.label, b.color = "n6_speedcam_case_body", CASE_COLOR
    return b


def lid_core():
    """The door without its crush ribs (the ribs are the only designed interference)."""
    # full outer footprint at the sides and bottom; the top edge stops under the roof
    l = slab(OUT_X, LID_Y1 - Y0, LID_T, r=CORNER_R, at=(X0, Y0, Z_CEIL))
    # chamfers, cut as wedges (OCC's chamfer refuses edges that end on the corner
    # fillets): top-back and top-front so the edge can rotate inside the pocket,
    # bottom-front as a drip arris
    cb, cf = LID_TOP_CHAMFER
    x0, x1 = X0 - 1.0, X0 + OUT_X + 1.0
    l = l - _wedge_along_x([(LID_Y1 + 0.01, Z_CEIL - 0.01), (LID_Y1 - cb, Z_CEIL - 0.01), (LID_Y1 + 0.01, Z_CEIL + cb)], x0, x1)
    l = l - _wedge_along_x([(LID_Y1 + 0.01, Z_TOP + 0.01), (LID_Y1 - cf, Z_TOP + 0.01), (LID_Y1 + 0.01, Z_TOP - cf)], x0, x1)
    c = LID_EDGE_CHAMFER
    l = l - _wedge_along_x([(Y0 - 0.01, Z_TOP + 0.01), (Y0 + c, Z_TOP + 0.01), (Y0 - 0.01, Z_TOP - c)], x0, x1)
    l = l + lid_lips()
    # optional screw blocks on the back, just above the floor wall; pilots open downward
    bw, bh, bd = LID_BLOCK
    for x in LID_SCREW_X:
        l = l + box((x - bw / 2, YI0 + LIP_GAP + LIP_T - 0.01, Z_CEIL - bd), (bw, bh, bd + 0.01))
        l = l - (Pos(x, YI0 - 1.0, LID_SCREW_Z) * Rot(-90, 0, 0) * Cylinder(
            LID_PILOT_D / 2, LID_PILOT_DEPTH + LIP_T + LIP_GAP + 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    # radome pocket, cut from the inside; the outer face stays flat
    px, py, pw, ph = POCKET
    l = l - box((px, py, Z_CEIL - 0.01), (pw, ph, (LID_T - RADOME_T) + 0.01))
    # radar bosses stand on the pocket floor; pilots stop at the radome
    for (x, y) in RADAR_HOLES:
        l = l + cyl((x, y), Z_RADAR_FACE, RADAR_BOSS_D, RADAR_STANDOFF + 0.01)
        l = l - cyl((x, y), Z_RADAR_FACE - 1, RADAR_PILOT_D, RADAR_STANDOFF + 1)
    # N6 bosses: full width down to the camera arm, then a narrower tip to the PCB
    for (x, y) in N6_MOUNT:
        l = l + cyl((x, y), N6_TIP_TOP_Z, N6_BOSS_D, Z_CEIL - N6_TIP_TOP_Z + 0.01)
        l = l + cyl((x, y), N6_PCB_T, N6_TIP_D, N6_TIP_TOP_Z - N6_PCB_T + 0.01)
        l = l - cyl((x, y), N6_PCB_T - 1, N6_PILOT_D, N6_PILOT_DEPTH + 1)
    # anti-bow pads over the N6's USB end
    for (x0, y0, dx, dy) in N6_PADS:
        l = l + box((x0, y0, N6_PCB_T + PAD_GAP), (dx, dy, Z_CEIL - N6_PCB_T - PAD_GAP + 0.01))
    # carrier bosses
    for (x, y) in CARRIER_HOLES:
        l = l + cyl((x, y), Z_CARRIER_PCB, CARRIER_BOSS_D, CARRIER_STANDOFF + 0.01)
        l = l - cyl((x, y), Z_CARRIER_PCB - 1, CARRIER_PILOT_D, CARRIER_PILOT_DEPTH + 1)
    # lens window: the barrel passes through and stands proud; a small outward
    # flare so the arris sheds instead of holding a ring of water on the barrel
    l = K.conical_window(l, CAM_C, Z_TOP, Z_CEIL, WINDOW_D, WINDOW_D_OUT)
    l.label, l.color = "n6_speedcam_case_lid", CASE_COLOR
    return l


def lid():
    l = lid_core() + crush_ribs()
    l.label, l.color = "n6_speedcam_case_lid", CASE_COLOR
    return l


# --- reference parts, placed in the case frame ----------------------------------------------
def n6_ref():
    p = N6.gen_step()
    p.label = "openmv_n6"
    return p


def radar_ref():
    p = Pos(RADAR_X0, RADAR_Y0, Z_RADAR_FACE) * RD.gen_step()
    p.label = "hlk_ld2415h"
    return p


def dfr_ref():
    p = Pos(DFR_X0, DFR_Y0, Z_DFR_PCB) * PM.gen_step()
    p.label = "dfr0535"
    return p


def carrier_ref():
    # component side toward the back: flip about X so local +Z becomes -Z, then
    # land the PCB's lid-side face on the boss ends
    p = Pos(CARRIER_X0, CARRIER_Y0 + CR.L, Z_CARRIER_PCB) * Rot(180, 0, 0) * CR.gen_step()
    p.label = "carrier"
    return p


def battery_ref():
    x0, y0, dx, dy, dz = BAT
    p = box((x0, y0, Z_IN_FLOOR), (dx, dy, dz), label="lipo_pouch_envelope", color=Color(0.85, 0.75, 0.2))
    return p


def refs():
    return [n6_ref(), radar_ref(), dfr_ref(), carrier_ref(), battery_ref()]
