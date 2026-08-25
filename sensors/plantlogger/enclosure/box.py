"""box.py — Bottom box of the 3-part FeatherS3D enclosure (FDM, printed open-side-up).

Coordinate frame = GLOBAL FRAME G from the interface spec:
  X=0 at the INNER face of the USB-end short wall
  Y=0 at the INNER face of the long wall nearest the board mounting-hole row
  Z=0 at the cavity floor top surface
Outer envelope therefore spans G X[-2.4, 70.4], Y[-2.4, 42.4], Z[-2.4, 18.5].

Features:
  - Rounded-corner shell: outer 72.8 x 44.8 x 20.9, R4.0 outside / R1.6 inside,
    walls and floor 2.4.
  - Four 45-deg right-triangle corner gussets (leg 3.5, height 6.5) = midplate
    standoffs; battery bay below them (usable height 6.5).
  - USB-C slot 13.0 x 8.0 R3.0 through the X=0 wall, centered G(Y=14.46, Z=13.46).
  - Pry notch 14.0 x 2.0 x 1.2 deep on the far short wall outer face, open to
    the rim, centered G Y=20.0 (exposes faceplate underside edge).
  - 0.5 x 45 deg lead-in chamfer on the cavity top inner edge (faceplate lip /
    midplate insertion friction surface).
  - 0.4 mm elephant-foot chamfer on the outer bottom perimeter edge.

Faceplate seat: NO rebate — plain square rim at Z=18.5; the faceplate cap rests
on the rim and its plug lip enters the chamfered cavity mouth.
"""

from build123d import (
    Align,
    Axis,
    Box,
    Plane,
    Polygon,
    Pos,
    RectangleRounded,
    chamfer,
    extrude,
)

# ----------------------------------------------------------------------------
# Parameters (mm) — all values from the interface spec; do not re-derive.
# ----------------------------------------------------------------------------
WALL = 2.4                  # side wall thickness
FLOOR = 2.4                 # floor thickness

CAV_L = 68.0                # cavity length (X)
CAV_W = 40.0                # cavity width  (Y)
CAV_D = 18.5                # cavity depth  (Z), rim at Z=18.5
CAV_R = 1.6                 # cavity inner corner radius

OUT_L = CAV_L + 2 * WALL    # 72.8 outer length
OUT_W = CAV_W + 2 * WALL    # 44.8 outer width
OUT_H = CAV_D + FLOOR       # 20.9 outer height (Z -2.4 .. 18.5)
OUT_R = 4.0                 # outer corner radius

# Corner gussets (midplate standoffs)
GUSSET_LEG = 3.5            # right-triangle leg along each wall
GUSSET_H = 6.5              # top face at Z=6.5 = midplate resting surface

# USB-C slot in the X=0 short wall
USB_W = 13.0                # slot width along Y
USB_H = 8.0                 # slot height along Z
USB_R = 3.0                 # slot corner radius
USB_CY = 14.46              # slot center, G Y
USB_CZ = 13.46              # slot center, G Z (15.86 above outer bottom face)

# Pry notch, far short wall (inner face X=68), outer surface, open to the rim
PRY_W = 14.0                # width along Y
PRY_H = 2.0                 # height along Z (Z 16.5 .. 18.5)
PRY_DEPTH = 1.2             # depth into the 2.4 wall
PRY_CY = 20.0               # centered on cavity width

# Chamfers
MOUTH_CHAMFER = 0.5         # 45-deg lead-in on cavity top inner edge
FOOT_CHAMFER = 0.4          # elephant-foot relief on outer bottom edge

OVER = 1.0                  # boolean overshoot

# Shared centroid of outer shell and cavity (uniform walls)
CX, CY = CAV_L / 2.0, CAV_W / 2.0  # (34.0, 20.0)

# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------


def _gusset(cx: float, cy: float, sx: float, sy: float):
    """45-deg right-triangle gusset prism at inner cavity corner (cx, cy).

    sx/sy are +1/-1 directions pointing INTO the cavity from that corner.
    Runs Z 0 -> GUSSET_H with a flat top (midplate resting surface).
    """
    pts = [
        (cx, cy),
        (cx + sx * GUSSET_LEG, cy),
        (cx, cy + sy * GUSSET_LEG),
    ]
    if sx * sy < 0:  # keep CCW winding so the face normal (extrude dir) is +Z
        pts[1], pts[2] = pts[2], pts[1]
    tri = Polygon(*pts, align=None)
    return extrude(tri, GUSSET_H)


def gen_step():
    # Outer rounded block: Z -2.4 .. 18.5
    outer = Pos(CX, CY, -FLOOR) * extrude(
        RectangleRounded(OUT_L, OUT_W, OUT_R), OUT_H
    )

    # Cavity: Z 0 .. rim (overshoot above rim for a clean cut)
    cavity = Pos(CX, CY, 0) * extrude(
        RectangleRounded(CAV_L, CAV_W, CAV_R), CAV_D + OVER
    )
    box = outer - cavity

    # Four corner gussets, fused to both walls at each inner corner
    box += _gusset(0.0, 0.0, +1, +1)
    box += _gusset(CAV_L, 0.0, -1, +1)
    box += _gusset(0.0, CAV_W, +1, -1)
    box += _gusset(CAV_L, CAV_W, -1, -1)

    # USB-C slot through the X=0 wall (wall spans X -2.4 .. 0).
    # Plane.YZ: local x -> global Y, local y -> global Z, normal -> +X.
    usb_slot = extrude(
        Plane.YZ.offset(-WALL - OVER)
        * Pos(USB_CY, USB_CZ)
        * RectangleRounded(USB_W, USB_H, USB_R),
        WALL + 2 * OVER,
    )
    box -= usb_slot

    # Pry notch: outer face of far short wall, Z 16.5 .. 18.5 (open to rim)
    notch = Box(
        PRY_DEPTH + OVER,
        PRY_W,
        PRY_H + OVER,
        align=(Align.MIN, Align.CENTER, Align.MIN),
    )
    box -= Pos(CAV_L + WALL - PRY_DEPTH, PRY_CY, CAV_D - PRY_H) * notch

    # Elephant-foot chamfer: outer bottom perimeter (lowest-Z edge group)
    bottom_edges = box.edges().group_by(Axis.Z)[0]
    box = chamfer(bottom_edges, FOOT_CHAMFER)

    # 0.5 x 45 lead-in on the cavity top INNER edge, full perimeter.
    # Inner rim edges lie at Z=18.5 with XY extents inside the cavity outline;
    # the outer rim perimeter and pry-notch rim edges fall outside this filter.
    tol = 0.05
    rim_inner = [
        e
        for e in box.edges()
        if (bb := e.bounding_box()).min.Z > CAV_D - tol
        and bb.min.X > -tol
        and bb.max.X < CAV_L + tol
        and bb.min.Y > -tol
        and bb.max.Y < CAV_W + tol
    ]
    box = chamfer(rim_inner, MOUTH_CHAMFER)

    box.label = "box"
    return box


if __name__ == "__main__":
    part = gen_step()
    print("bbox:", part.bounding_box())
