"""OpenMV Cam N6 + 1S LiPo — 3-part printed case (geometry library).

Three printed parts:

  front_cup   outer shell over the lens/component side of the N6.  Straight
              bore for the board, an internal shoulder that the cam plate
              bears against, then a deeper socket that receives the back
              cup.  Lens barrel stands proud through the face.
  cam_plate   flat plate the N6 bolts to with 2 x M2.5 through its Ø2.80
              mounting holes.  Battery sits on its back face.  Sandwiched
              between the front cup shoulder and the back cup lip.
  back_cup    lip plugs into the front cup socket (0.15/side + crush ribs),
              pushes the plate up against the shoulder, and encloses the
              battery bay.

Coordinate frame — identical to ref/openmv-n6.py:
  origin at the N6 PCB bottom-left corner, on the PCB *bottom* face.
  +X 35.56 board width, +Y 44.45 board length (lens end at +Y),
  +Z optical axis.  Z = 0 is the PCB bottom face.

Fit values come from the project-local fits.py (ex-mywarehouse, now inlined).
"""

from build123d import *  # noqa: F403
import fits

# ---------------------------------------------------------------------------
# N6 board envelope (measured — see ref/DIMENSIONS.md)
# ---------------------------------------------------------------------------
BOARD_X0, BOARD_X1 = -0.102, 35.712   # incl. camera daughter-board arm overhang
BOARD_Y0, BOARD_Y1 = 0.000, 45.080    # incl. side-actuated button overhang
Z_TAIL = -3.000                       # header tails below the PCB (assumed 3.0)
Z_LOCKRING_TOP = 22.850               # M12 lock-ring top, above PCB bottom
Z_LENS_TIP = 31.250
LENS_AXIS = (17.810, 36.255)
LENS_BARREL_D = 14.000
MOUNT_HOLES = [(3.048, 41.402), (32.512, 41.402)]   # Ø2.80, 29.464 apart

# ---------------------------------------------------------------------------
# Fits and print rules (mywarehouse tolerances.md / printing.md)
# ---------------------------------------------------------------------------
WALL = 2.40                 # enclosure structure wall = 6 perimeters
LIP_WALL = 1.60
LIP_GAP = fits.LIP_GAP      # 0.15/side, press-fit lip into mouth
LIP_RIB_PROUD = fits.LIP_RIB_PROUD   # 0.25 crush ribs -> 0.10/side net
PLATE_GAP = fits.PLATE_SLIDE_GAP     # 0.20/side, plate in socket
EFOOT = fits.ELEPHANT_FOOT  # 0.40 x 45 deg on bed-contact perimeters
LEADIN = 0.60               # 45 deg lead-in on friction entries

BOARD_CLR = 0.60            # board -> front cup bore, per side
CARD_CH = 3.60              # +X bore side runs this wide instead (microSD card
                            # channel — see the note above SD_YC below)
SHOULDER = 1.20             # ledge width the cam plate bears on
WIRE_CH = 8.00              # battery-lead channel at the USB (-Y) end: JST-PH
                            # plug + lead loop need real room (rev C, was 4.00)

# ---------------------------------------------------------------------------
# Battery bay — 30 x 40 x 6 pouch assumed; bay is board-sized, per
# tolerances "bay height = max thickness + ~1.5 swell/wire room".
# ---------------------------------------------------------------------------
BAT_T = 6.00
BAT_SWELL = 1.50
BAY_DEPTH = BAT_T + BAT_SWELL       # 7.50

# ---------------------------------------------------------------------------
# Z stack (all relative to the PCB bottom face)
# ---------------------------------------------------------------------------
Z_SHOULDER = -3.50                          # cam plate front face
PLATE_T = 3.00
Z_PLATE_BOT = Z_SHOULDER - PLATE_T          # -6.50, back cup lip top face
Z_SEAM = Z_PLATE_BOT - BAY_DEPTH            # -14.00, front cup rim
Z_BACK_OUT = Z_SEAM - WALL                  # -16.40, case back face
Z_CEIL = Z_LOCKRING_TOP + 0.65              # 23.50, front cup inner ceiling
Z_FRONT_OUT = Z_CEIL + WALL                 # 25.90, case front face

# Rev B: +2.00 not +1.00 — print 1 of the RT1062 case (same M12 lens family)
# bound on the knurled focus ring, ~Ø15.5 actual vs Ø14.0 in OpenMV's GLB.
# Ø16.00 clears the knurl 0.25/side and still keeps the Ø16.20 lock ring captive.
LENS_HOLE_D = LENS_BARREL_D + 2.00          # 16.00, barrel stands proud

# ---------------------------------------------------------------------------
# XY rectangles, built outward from the board envelope
# ---------------------------------------------------------------------------
CAV_X0, CAV_X1 = BOARD_X0 - BOARD_CLR, BOARD_X1 + CARD_CH
CAV_Y0, CAV_Y1 = BOARD_Y0 - WIRE_CH, BOARD_Y1 + BOARD_CLR

SOCK_X0, SOCK_X1 = CAV_X0 - SHOULDER, CAV_X1 + SHOULDER
SOCK_Y0, SOCK_Y1 = CAV_Y0 - SHOULDER, CAV_Y1 + SHOULDER

OUT_X0, OUT_X1 = SOCK_X0 - WALL, SOCK_X1 + WALL
OUT_Y0, OUT_Y1 = SOCK_Y0 - WALL, SOCK_Y1 + WALL

PLATE_X0, PLATE_X1 = SOCK_X0 + PLATE_GAP, SOCK_X1 - PLATE_GAP
PLATE_Y0, PLATE_Y1 = SOCK_Y0 + PLATE_GAP, SOCK_Y1 - PLATE_GAP

LIP_X0, LIP_X1 = SOCK_X0 + LIP_GAP, SOCK_X1 - LIP_GAP
LIP_Y0, LIP_Y1 = SOCK_Y0 + LIP_GAP, SOCK_Y1 - LIP_GAP

BAY_X0, BAY_X1 = LIP_X0 + LIP_WALL, LIP_X1 - LIP_WALL
BAY_Y0, BAY_Y1 = LIP_Y0 + LIP_WALL, LIP_Y1 - LIP_WALL

R_OUT = 6.00
R_SOCK = R_OUT - WALL                # 3.60
# Cavity radius is set independently of the socket (the shoulder simply runs
# wider at the corners).  A 2.40 corner would leave the square PCB corners
# only 0.06 clear of the bore; 1.20 keeps them above 0.60 all round.
R_CAV = 1.20
R_LIP = R_SOCK - LIP_GAP             # 3.45
R_BAY = R_LIP - LIP_WALL             # 1.85
R_PLATE = R_SOCK + 0.40              # 4.00 > socket radius: corners never bind

CX = (OUT_X0 + OUT_X1) / 2
CY = (OUT_Y0 + OUT_Y1) / 2

# ---------------------------------------------------------------------------
# M2.5 board screws into thread-forming bosses
# ---------------------------------------------------------------------------
BOSS_D = 5.50
BOSS_PILOT_D = 2.10
# Blind pilot, measured down from the PCB bottom face.  5.30 leaves a 1.20
# floor under the hole (printing.md: no feature floor below 1.20) and takes
# an M2.5 x 6 with 4.70 of thread engagement.
BOSS_PILOT_DEPTH = 5.30

PAD = [(5.4, 0.5, 7.4, 3.0), (28.7, 0.5, 30.7, 3.0)]   # anti-bow pads, USB end
PAD_TOP_Z = -0.30                    # 0.3 below the PCB: catches bow, no preload

WIRE_SLOT = (5.0, -7.5, 16.0, -2.0)  # battery leads through the plate; big
                                     # enough to pass a mated JST-PH plug

# ---------------------------------------------------------------------------
# microSD card channel — the +X cavity side runs CARD_CH wide for its full
# height instead of BOARD_CLR.  The socket (x 23.62..35.02, bottom side)
# opens toward the +X board edge; a seated card's tail reaches
# x = 35.02 + (15.0 - 11.40) = 38.62, 3.06 past the PCB edge, so the whole
# board+plate must slide in with that overhang.  Rev D: full-height channel
# (case is 3.0 wider), wall closed — the card stays in for the camera's life
# and is not reachable from outside (rev C's through-slot removed).
# ---------------------------------------------------------------------------
# (CARD_CH = 3.60 lives next to BOARD_CLR: BOARD_X1 + 3.60 = 39.31 > 38.62 + 0.60)
SD_YC = 27.125                       # card centreline (socket centre), for mock

# ---------------------------------------------------------------------------
# USB-C port — through the -Y wall.  Receptacle x 18.88..28.46,
# z 0.45..4.61; opening passes a plug overmold up to 15.0 x 9.5.
# ---------------------------------------------------------------------------
USB_XC, USB_ZC = 23.67, 2.53
USB_W, USB_H = 15.00, 9.50

# ---------------------------------------------------------------------------
# Vents — bottom (-Y) wall of the front cup
# ---------------------------------------------------------------------------
VENT_W, VENT_H = 2.40, 12.00
VENT_ZC = 8.00
# Rev C: only the two vents clear of the USB port remain; the port's own
# 15.0 x 9.5 opening supplies more free area than the three slots it displaced.
VENT_XC = [CX + k * 6.0 for k in (-2, -1)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sk(w, l, r, z):
    """Rounded-rectangle sketch on a plane at height z, centred on the origin."""
    return Plane.XY.offset(z) * RectangleRounded(w, l, r)


def prism(x0, y0, x1, y1, r, z0, z1, cb=0.0, ct=0.0):
    """Rounded-rect prism spanning z0..z1, optional 45 deg chamfer each end."""
    w, l = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    zb, zt = z0 + cb, z1 - ct
    solid = Pos(cx, cy) * extrude(_sk(w, l, r, zb), amount=zt - zb)
    if cb:
        solid += Pos(cx, cy) * loft(
            [_sk(w - 2 * cb, l - 2 * cb, max(r - cb, 0.05), z0), _sk(w, l, r, zb)]
        )
    if ct:
        solid += Pos(cx, cy) * loft(
            [_sk(w, l, r, zt), _sk(w - 2 * ct, l - 2 * ct, max(r - ct, 0.05), z1)]
        )
    return solid


def flare_down(x0, y0, x1, y1, r, z0, c):
    """45 deg lead-in cone for a bore: c oversize at z0, nominal at z0 + c."""
    w, l = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return Pos(cx, cy) * loft(
        [_sk(w + 2 * c, l + 2 * c, r + c, z0), _sk(w, l, r, z0 + c)]
    )


def slot_y(xc, zc, w, h, y0, y1, r=None):
    """Rounded slot through a -Y/+Y wall, w along X, h along Z (stadium default)."""
    sk = RectangleRounded(w, h, min(w, h) / 2 - 0.001 if r is None else r)
    return Pos(xc, y0, zc) * Rot(-90, 0, 0) * extrude(sk, amount=y1 - y0)


def slot_x(yc, zc, w, h, x0, x1):
    """Stadium slot through a -X/+X wall, w along Y, h along Z."""
    sk = RectangleRounded(h, w, min(w, h) / 2 - 0.001)
    return Pos(x0, yc, zc) * Rot(0, 90, 0) * extrude(sk, amount=x1 - x0)


def box_at(x0, y0, z0, dx, dy, dz):
    return Pos(x0, y0, z0) * Box(dx, dy, dz, align=(Align.MIN,) * 3)


def cyl_at(cx, cy, z0, d, h):
    return Pos(cx, cy, z0) * Cylinder(
        d / 2, h, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------
def front_cup():
    """Prints face-down (Z_FRONT_OUT on the bed).  No supports."""
    part = prism(OUT_X0, OUT_Y0, OUT_X1, OUT_Y1, R_OUT, Z_SEAM, Z_FRONT_OUT,
                 ct=EFOOT)

    # board cavity, then the deeper socket; the step between them is the
    # 1.20 shoulder the cam plate is clamped against
    part -= prism(CAV_X0, CAV_Y0, CAV_X1, CAV_Y1, R_CAV, Z_SHOULDER, Z_CEIL)
    part -= prism(SOCK_X0, SOCK_Y0, SOCK_X1, SOCK_Y1, R_SOCK,
                  Z_SEAM - 1.0, Z_SHOULDER)
    part -= flare_down(SOCK_X0, SOCK_Y0, SOCK_X1, SOCK_Y1, R_SOCK,
                       Z_SEAM, LEADIN)

    # lens aperture: barrel stands proud, lock ring stays captive inside
    part -= cyl_at(*LENS_AXIS, Z_CEIL - 1.0, LENS_HOLE_D, WALL + 2.0)
    part -= Pos(*LENS_AXIS, Z_FRONT_OUT - EFOOT) * Cone(
        bottom_radius=LENS_HOLE_D / 2,
        top_radius=LENS_HOLE_D / 2 + EFOOT,
        height=EFOOT,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    for xc in VENT_XC:
        part -= slot_y(xc, VENT_ZC, VENT_W, VENT_H, OUT_Y0 - 1.0, CAV_Y0 + 1.0)

    # USB-C port through the -Y wall (r=2.5 keeps the corners open for a
    # square-shouldered plug overmold)
    part -= slot_y(USB_XC, USB_ZC, USB_W, USB_H, OUT_Y0 - 1.0, CAV_Y0 + 1.0,
                   r=2.5)

    part.label = "front_cup"
    return part


def cam_plate():
    """Prints flat, bosses up (Z_PLATE_BOT on the bed).  No supports."""
    part = prism(PLATE_X0, PLATE_Y0, PLATE_X1, PLATE_Y1, R_PLATE,
                 Z_PLATE_BOT, Z_SHOULDER, cb=EFOOT)

    for cx, cy in MOUNT_HOLES:                      # M2.5 bosses up to the PCB
        part += cyl_at(cx, cy, Z_SHOULDER, BOSS_D, -Z_SHOULDER)
    for x0, y0, x1, y1 in PAD:                      # anti-bow pads at the USB end
        part += box_at(x0, y0, Z_SHOULDER, x1 - x0, y1 - y0,
                       PAD_TOP_Z - Z_SHOULDER)
    for cx, cy in MOUNT_HOLES:
        part -= cyl_at(cx, cy, -BOSS_PILOT_DEPTH, BOSS_PILOT_D,
                       BOSS_PILOT_DEPTH + 1.0)

    x0, y0, x1, y1 = WIRE_SLOT                      # battery leads to the JST
    part -= Pos((x0 + x1) / 2, (y0 + y1) / 2, Z_PLATE_BOT - 1.0) * extrude(
        RectangleRounded(x1 - x0, y1 - y0, (y1 - y0) / 2 - 0.001),
        amount=PLATE_T + 2.0,
    )

    part.label = "cam_plate"
    return part


def back_cup():
    """Prints back-face-down (Z_BACK_OUT on the bed).  No supports."""
    part = prism(OUT_X0, OUT_Y0, OUT_X1, OUT_Y1, R_OUT, Z_BACK_OUT, Z_SEAM,
                 cb=EFOOT)
    part += prism(LIP_X0, LIP_Y0, LIP_X1, LIP_Y1, R_LIP, Z_SEAM, Z_PLATE_BOT,
                  ct=LEADIN)
    part -= prism(BAY_X0, BAY_Y0, BAY_X1, BAY_Y1, R_BAY, Z_SEAM, Z_PLATE_BOT)

    # crush ribs carry the interference; bulk lip surfaces keep 0.15 clearance
    rib_h = (Z_PLATE_BOT - LEADIN - 0.5) - Z_SEAM
    rib = fits.edge_crush_rib(rib_h, length=6.0, proud=LIP_RIB_PROUD)
    for yc in (CY - 12.0, CY + 12.0):
        part += Pos(LIP_X0, yc, Z_SEAM) * Rot(0, 0, 90) * rib
        part += Pos(LIP_X1, yc, Z_SEAM) * Rot(0, 0, -90) * rib
    part += Pos(CX, LIP_Y0, Z_SEAM) * Rot(0, 0, 180) * rib
    part += Pos(CX, LIP_Y1, Z_SEAM) * rib

    part.label = "back_cup"
    return part


def battery_mock():
    """30 x 40 x 6 pouch, seated against the plate back face — reference only."""
    part = box_at(CX - 15.0, (BAY_Y0 + BAY_Y1) / 2 - 20.0, Z_PLATE_BOT - BAT_T,
                  30.0, 40.0, BAT_T)
    part.label = "battery_30x40x6_mock"
    return part


def sd_card_mock():
    """microSD seated in the socket: 11 x 15 x 1.0, tail proud of the PCB edge."""
    part = box_at(38.62 - 15.0, SD_YC - 5.5, -1.36, 15.0, 11.0, 1.0)
    part.label = "microsd_card_mock"
    return part


def usb_plug_mock():
    """Seated USB-C plug: shell at the receptacle mouth + 12 x 7 overmold
    running out through the port.  Reference for clearance checks only."""
    part = box_at(USB_XC - 4.47, -1.0, USB_ZC - 1.6, 8.94, 7.5, 3.2)
    part += box_at(USB_XC - 6.0, OUT_Y0 - 12.0, USB_ZC - 3.5,
                   12.0, (-1.5) - (OUT_Y0 - 12.0), 7.0)
    part.label = "usb_plug_mock"
    return part


def battery_cable_mock():
    """JST plug + lead loop: over the -Y board edge, down the wire channel and
    through the plate slot into the bay.  Reference for clearance checks only."""
    part = box_at(6.0, -7.0, 6.5, 8.0, 15.0, 3.0)          # loop over the JST
    part += box_at(6.5, -6.8, Z_PLATE_BOT - BAT_T,          # drop into the bay
                   7.0, 4.1, 6.5 - (Z_PLATE_BOT - BAT_T))
    part.label = "battery_cable_mock"
    return part
