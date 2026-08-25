"""
midplate.py - Removable friction-fit mid plate for the FeatherS3D 3-part enclosure.

Part 2 of 3 (box / midplate / faceplate). FDM, printed flat, posts up, no supports.

Coordinate convention (LOCAL plate frame, per interface spec):
  Origin = plate corner that seats at global G(0.2, 0.2); local = G - 0.2 in X and Y.
  Z = 0 at the plate UNDERSIDE (rests on box gusset tops at G Z=6.5; plate top at G Z=8.9).
  Plate spans local X [0, 67.6], Y [0, 39.6], Z [0, 2.4].

Function:
  - Drops into the 68.0 x 40.0 box cavity (0.20 mm clearance/side) and friction-fits
    via 6 crush ribs 0.30 mm proud (net 0.10 mm interference per side at ribs).
  - Rests on the four 45-deg corner gussets (tops at G Z=6.5); battery lives below.
  - FeatherS3D push-fits onto 3 ribbed posts through its 3 mounting holes
    (holes 2.50/2.50/2.54 mm; pin 2.3 nominal, 2.70 effective over ribs).
  - Battery-wire U-notch on the +Y edge; 10 mm finger hole to lift the plate out.
"""

from build123d import *

# ---------------------------------------------------------------------------
# Parameters (mm) - all mating dims fixed by the enclosure interface spec
# ---------------------------------------------------------------------------

# Plate outline (cavity 68.0 x 40.0 minus 0.20 clearance per side)
PLATE_X = 67.6
PLATE_Y = 39.6
PLATE_T = 2.4
PLATE_CORNER_R = 2.0          # > cavity inner R1.6, drops in freely
BOTTOM_CHAMFER = 0.4          # insertion lead-in + elephant-foot, bottom perimeter

# Edge crush ribs (friction fit against cavity walls)
RIB_LEN = 6.0                 # along-edge length at the plate face
RIB_PROUD = 0.30              # protrusion beyond plate edge face
RIB_EMBED = 0.4               # burial into the plate for a solid fuse
RIB_END_TAPER = RIB_PROUD     # 45-deg tapered ends (crest length = 6.0 - 2*0.30)
RIB_BOTTOM_CHAMFER = 0.25     # lead-in on rib bottom crest edge
RIB_LONG_FACE_XC = [21.0, 50.0]   # rib centers on Y=0 and Y=39.6 faces
# (21.0, not 17.0: rib base spans center +/-3.0; at 21.0 it clears the battery-wire
#  notch span 8.6-16.6 so the notch keeps its full 8.0 mm clear opening)
RIB_SHORT_FACE_YC = 19.8          # one rib centered on each of X=0 / X=67.6 faces

# Push-fit board posts - EXACT copies of measured FeatherS3D hole coords
# board holes (2.54,2.54), (2.54,20.32), (48.72,2.54) + board origin at plate-local (1.8, 2.8)
POST_CENTERS = [(4.34, 5.34), (4.34, 23.12), (50.52, 5.34)]
SHOULDER_D = 4.8              # < 5.08 pad annular-ring keep-out
SHOULDER_H = 2.0              # sets PCB bottom at G Z = 8.9 + 2.0 = 10.90
PIN_D = 2.3                   # nominal, clears 2.50/2.54 holes by >= 0.2
PIN_LEN = 1.9                 # above shoulder (tip at G Z = 12.8)
PIN_TIP_D = 1.8               # lead-in cone end diameter
PIN_TIP_LEN = 0.8             # lead-in cone length (final 0.8 of pin)
PIN_RIB_N = 3                 # axial crush ribs, equally spaced
PIN_RIB_W = 0.4
PIN_RIB_PROUD = 0.20          # -> 2.70 effective dia; 0.16-0.20 net interference
PIN_RIB_LEADIN = 0.20         # 45-deg taper at rib top so the hole starts easily

# Battery-wire U-notch on the +Y edge (local Y = 39.6, against the Y=40 wall)
NOTCH_CX = 12.6               # = G X 12.8, centered on board battery-JST span
NOTCH_W = 8.0
NOTCH_D = 5.0
NOTCH_R = 1.0                 # internal corner radius

# Finger/lift hole in the board-free zone
FINGER_HOLE_D = 10.0
FINGER_HOLE_C = (59.8, 19.8)  # = G(60, 20)

CUT_OVER = 1.0                # boolean overshoot


def _edge_rib() -> Part:
    """One crush rib in canonical pose: crest facing +Y, centered on X=0,
    base plane at Y=0 (the plate edge face), full plate thickness in Z."""
    half = RIB_LEN / 2
    pts = [
        (-half, -RIB_EMBED),
        (half, -RIB_EMBED),
        (half, 0.0),
        (half - RIB_END_TAPER, RIB_PROUD),     # 45-deg tapered end
        (-(half - RIB_END_TAPER), RIB_PROUD),  # 45-deg tapered end
        (-half, 0.0),
    ]
    rib = extrude(Polygon(*pts, align=None), PLATE_T)
    # Lead-in chamfer on the bottom crest edge (first contact when inserting)
    crest_bottom = [
        e for e in rib.edges()
        if abs(e.center().Z) < 1e-6 and abs(e.center().Y - RIB_PROUD) < 1e-6
    ]
    return chamfer(crest_bottom, RIB_BOTTOM_CHAMFER)


def _post() -> Part:
    """One push-fit post in canonical pose: axis +Z, shoulder base at Z=0
    (gets placed on the plate top face)."""
    shoulder = extrude(Circle(SHOULDER_D / 2), SHOULDER_H)
    straight_len = PIN_LEN - PIN_TIP_LEN                      # 1.1
    pin = extrude(Circle(PIN_D / 2), SHOULDER_H + straight_len)  # overlaps shoulder
    tip = Pos(0, 0, SHOULDER_H + straight_len) * Cone(
        bottom_radius=PIN_D / 2,
        top_radius=PIN_TIP_D / 2,
        height=PIN_TIP_LEN,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    # Axial crush rib profile in the radial(X)-axial(Z) plane, extruded 0.4 wide.
    r_in = PIN_D / 2 - 0.25                # buried into the pin
    r_crest = PIN_D / 2 + PIN_RIB_PROUD    # 1.35 -> 2.70 effective dia
    z0 = SHOULDER_H                        # rib starts at shoulder top
    z1 = SHOULDER_H + straight_len         # rib ends where the tip cone starts
    prof = Plane.XZ * Polygon(
        (r_in, z0),
        (r_crest, z0),
        (r_crest, z1 - PIN_RIB_LEADIN),
        (PIN_D / 2, z1),                   # 45-deg lead-in taper at rib top
        (r_in, z1),
        align=None,
    )
    rib = extrude(prof, PIN_RIB_W / 2, both=True)
    ribs = [Rot(0, 0, a) * rib for a in (0, 120, 240)]
    post = shoulder + pin + tip
    for r in ribs:
        post += r
    return post


def gen_step():
    # --- Base plate -------------------------------------------------------
    plate_sk = Pos(PLATE_X / 2, PLATE_Y / 2) * RectangleRounded(
        PLATE_X, PLATE_Y, PLATE_CORNER_R
    )
    plate = extrude(plate_sk, PLATE_T)

    # Bottom perimeter chamfer (insertion lead-in + elephant foot)
    plate = chamfer(plate.edges().group_by(Axis.Z)[0], BOTTOM_CHAMFER)

    # --- Battery-wire U-notch on the +Y edge ------------------------------
    notch_sk = Pos(NOTCH_CX, PLATE_Y - NOTCH_D + (NOTCH_D + CUT_OVER) / 2) * Rectangle(
        NOTCH_W, NOTCH_D + CUT_OVER
    )
    inner_corners = notch_sk.vertices().group_by(Axis.Y)[0]
    notch_sk = fillet(inner_corners, NOTCH_R)
    notch = extrude(Pos(0, 0, -CUT_OVER) * notch_sk, PLATE_T + 2 * CUT_OVER)
    plate -= notch

    # --- Finger/lift hole ---------------------------------------------------
    finger = Pos(FINGER_HOLE_C[0], FINGER_HOLE_C[1], -CUT_OVER) * extrude(
        Circle(FINGER_HOLE_D / 2), PLATE_T + 2 * CUT_OVER
    )
    plate -= finger

    # --- Edge crush ribs ----------------------------------------------------
    rib = _edge_rib()
    for xc in RIB_LONG_FACE_XC:
        plate += Pos(xc, 0, 0) * Rot(0, 0, 180) * rib          # Y=0 face, crest -Y
        plate += Pos(xc, PLATE_Y, 0) * rib                     # Y=39.6 face, crest +Y
    plate += Pos(0, RIB_SHORT_FACE_YC, 0) * Rot(0, 0, 90) * rib    # X=0 face, crest -X
    plate += Pos(PLATE_X, RIB_SHORT_FACE_YC, 0) * Rot(0, 0, -90) * rib  # X=67.6, crest +X

    # --- Push-fit board posts ------------------------------------------------
    post = _post()
    for cx, cy in POST_CENTERS:
        plate += Pos(cx, cy, PLATE_T) * post

    plate.label = "midplate"
    return plate
