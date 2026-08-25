"""
faceplate.py — friction-lip top faceplate for the FeatherS3D 3-part enclosure.

Part 3 of 3 (box / midplate / faceplate). Solid cap except the sensor-wire
exit hole; plug lip friction-fits into the box cavity; pry access is provided
by the notch in the box rim (no pry feature on this part).

Coordinate convention (matches the interface spec's faceplate local frame):
  - Local X/Y: local = GLOBAL-G + 2.4  (local origin = outer corner that maps
    to G(-2.4, -2.4); panel spans local X 0..72.8, Y 0..44.8).
  - Local Z: 0 at the panel UNDERSIDE = box-rim contact plane (G Z=18.5).
    Panel top at +2.4 (G 20.9); lip bottom at -4.0 (G 14.5).
  - Print orientation: flat, top-face-down (lip up). The 0.4 chamfer on the
    top outer perimeter doubles as the elephant-foot chamfer.

All dims mm. Interface (mating) dimensions are copied from the spec verbatim
and must not be changed independently of the box part.
"""

from math import atan2, degrees

from build123d import *

# ---------------- top panel (cap) ----------------
PANEL_X = 72.8            # = box outer footprint X
PANEL_Y = 44.8            # = box outer footprint Y
PANEL_T = 2.4             # cap thickness; rests on box rim at G Z=18.5
PANEL_R = 4.0             # outer corner radius = box outer R
PANEL_CX = PANEL_X / 2    # 36.4
PANEL_CY = PANEL_Y / 2    # 22.4
TOP_CHAMFER = 0.4         # top outer perimeter (also elephant-foot on bed face)

# ---------------- plug lip (friction ring into cavity) ----------------
LIP_X = 67.7              # cavity 68.0 - 2*0.15 clearance
LIP_Y = 39.7              # cavity 40.0 - 2*0.15 clearance
LIP_DEPTH = 4.0           # lip bottom at local Z -4.0 (G 14.5)
LIP_R = 1.45              # outer corner radius (cavity R1.6 - 0.15)
LIP_WALL = 2.0            # ring wall width
LIP_R_IN = 0.75           # inner corner radius (non-mating, small)
LIP_X0 = PANEL_CX - LIP_X / 2   # 2.55  (local span X 2.55..70.25)
LIP_X1 = PANEL_CX + LIP_X / 2   # 70.25
LIP_Y0 = PANEL_CY - LIP_Y / 2   # 2.55  (local span Y 2.55..42.25)
LIP_Y1 = PANEL_CY + LIP_Y / 2   # 42.25
LIP_LEADIN = 0.5          # 45-deg lead-in chamfer around lip bottom outer edge

# ---------------- USB lip relief (mandatory) ----------------
# Lip segment on the USB wall (local X=2.55 face) removed full depth so the
# lip cannot intrude into the USB slot opening / cable overmold.
USB_RELIEF_W = 16.0
USB_RELIEF_YC = 16.86               # center local Y (= G 14.46)
USB_RELIEF_Y0 = USB_RELIEF_YC - USB_RELIEF_W / 2   # 8.86
USB_RELIEF_Y1 = USB_RELIEF_YC + USB_RELIEF_W / 2   # 24.86

# ---------------- friction crush ribs (5) ----------------
# Vertical ribs on lip outer faces, full lip depth, 45-deg tapered ends.
# Net fit: 0.15 clearance - 0.25 proud = 0.10 interference/side at ribs.
RIB_LEN = 6.0             # along-face length at the root
RIB_PROUD = 0.25          # radial protrusion (crest length = 6.0 - 2*0.25 = 5.5)
RIB_EMBED = 0.30          # root embedded into lip wall for a clean fuse
RIB_X_CENTERS = (19.4, 52.4)   # on both long faces (Y=2.55 and Y=42.25)
RIB_FARX_YC = 22.4             # single rib on far short face (X=70.25)
# No rib on the USB-relief side (X=2.55 face).

# ---------------- sensor hole ----------------
SENSOR_HOLE_D = 12.0                 # passes JST-PH PHR-4 plug (11.5 min)
SENSOR_HOLE_XY = (26.73, 22.42)      # local; = G(24.33, 20.02), above STEMMA #1
HOLE_CHAMFER = 0.4                   # both ends of the hole

# Light slot: over the ALS-PT19 ambient light sensor AND the blue status
# LED (adjacent on the board) - lets daylight reach the sensor and makes
# the heartbeat blink visible. Ends = component centers measured from the
# vendor STEP: ALS board(35.15,18.17), LED board(38.04,18.00).
LIGHT_SLOT_A = (39.55, 23.57)        # local; = G(37.15, 21.17), ALS center
LIGHT_SLOT_B = (42.44, 23.40)        # local; = G(40.04, 21.00), blue LED
LIGHT_SLOT_D = 5.0                   # slot width (stadium diameter)
LIGHT_SLOT_TOP_CHAMFER = 0.8         # wide top lead-in -> bigger sky acceptance cone

# ---------------- orientation marking ----------------
# Engraved triangle on the top face near the USB-relief edge, apex pointing
# at local X=0 (the 'USB' side). Mandatory: plate inserts 180-deg rotated but
# then blocks the USB slot.
MARK_DEPTH = 0.4
MARK_APEX = (5.0, PANEL_CY)                    # apex toward X=0
MARK_BASE = ((10.0, PANEL_CY - 2.5), (10.0, PANEL_CY + 2.5))


def _rib_pts(face: str, c: float):
    """Trapezoid (XY) for one crush rib. face in {'y0','y1','x1'}, c = center
    along the face. Root width RIB_LEN at the lip face, crest width
    RIB_LEN - 2*RIB_PROUD (45-deg tapered ends), embedded RIB_EMBED inward."""
    if face == "y0":    # lip face Y=LIP_Y0, protrudes -Y
        return [(c - 3.0, LIP_Y0 + RIB_EMBED), (c + 3.0, LIP_Y0 + RIB_EMBED),
                (c + 3.0, LIP_Y0), (c + 2.75, LIP_Y0 - RIB_PROUD),
                (c - 2.75, LIP_Y0 - RIB_PROUD), (c - 3.0, LIP_Y0)]
    if face == "y1":    # lip face Y=LIP_Y1, protrudes +Y
        return [(c - 3.0, LIP_Y1 - RIB_EMBED), (c + 3.0, LIP_Y1 - RIB_EMBED),
                (c + 3.0, LIP_Y1), (c + 2.75, LIP_Y1 + RIB_PROUD),
                (c - 2.75, LIP_Y1 + RIB_PROUD), (c - 3.0, LIP_Y1)]
    # 'x1': lip face X=LIP_X1, protrudes +X
    return [(LIP_X1 - RIB_EMBED, c - 3.0), (LIP_X1 - RIB_EMBED, c + 3.0),
            (LIP_X1, c + 3.0), (LIP_X1 + RIB_PROUD, c + 2.75),
            (LIP_X1 + RIB_PROUD, c - 2.75), (LIP_X1, c - 3.0)]


def gen_step():
    with BuildPart() as fp:
        # --- top panel ---
        with BuildSketch(Plane.XY):
            with Locations((PANEL_CX, PANEL_CY)):
                RectangleRounded(PANEL_X, PANEL_Y, PANEL_R)
        extrude(amount=PANEL_T)

        # --- plug lip ring, descending into the cavity ---
        with BuildSketch(Plane.XY):
            with Locations((PANEL_CX, PANEL_CY)):
                RectangleRounded(LIP_X, LIP_Y, LIP_R)
                RectangleRounded(LIP_X - 2 * LIP_WALL, LIP_Y - 2 * LIP_WALL,
                                 LIP_R_IN, mode=Mode.SUBTRACT)
        extrude(amount=-LIP_DEPTH)

        # --- USB lip relief: remove lip over local Y 8.86..24.86 on X=2.55 face ---
        with BuildSketch(Plane.XY):
            with Locations(((1.5 + (LIP_X0 + LIP_WALL + 0.5)) / 2,
                            USB_RELIEF_YC)):
                Rectangle((LIP_X0 + LIP_WALL + 0.5) - 1.5, USB_RELIEF_W)
        extrude(amount=-(LIP_DEPTH + 0.3), mode=Mode.SUBTRACT)

        # --- crush ribs (full lip depth; bottoms tapered by the wedge cuts) ---
        with BuildSketch(Plane.XY):
            for xc in RIB_X_CENTERS:
                Polygon(*_rib_pts("y0", xc))
                Polygon(*_rib_pts("y1", xc))
            Polygon(*_rib_pts("x1", RIB_FARX_YC))
        extrude(amount=-LIP_DEPTH)

        # --- 45-deg lead-in: four wedge cuts along the lip bottom outer edge.
        # Each ramp plane starts LIP_LEADIN inside the lip face at Z=-4.0 and
        # rises at 45 deg past the rib crests, so lip AND rib bottoms get one
        # continuous chamfer (>=0.5 on the lip face, full taper over ribs).
        with BuildSketch(Plane.YZ.offset(-1.0)):     # (u,v) -> (Y,Z); cuts run along X
            Polygon((LIP_Y0 + LIP_LEADIN, -LIP_DEPTH), (LIP_Y0 - 1.45, -LIP_DEPTH + 1.95),
                    (LIP_Y0 - 1.45, -LIP_DEPTH - 0.6), (LIP_Y0 + LIP_LEADIN, -LIP_DEPTH - 0.6))
            Polygon((LIP_Y1 - LIP_LEADIN, -LIP_DEPTH), (LIP_Y1 + 1.45, -LIP_DEPTH + 1.95),
                    (LIP_Y1 + 1.45, -LIP_DEPTH - 0.6), (LIP_Y1 - LIP_LEADIN, -LIP_DEPTH - 0.6))
        extrude(amount=PANEL_X + 2.0, mode=Mode.SUBTRACT)
        with BuildSketch(Plane.XZ.offset(-(PANEL_Y + 1.0))):  # (u,v) -> (X,Z); cuts run along -Y
            Polygon((LIP_X0 + LIP_LEADIN, -LIP_DEPTH), (LIP_X0 - 1.45, -LIP_DEPTH + 1.95),
                    (LIP_X0 - 1.45, -LIP_DEPTH - 0.6), (LIP_X0 + LIP_LEADIN, -LIP_DEPTH - 0.6))
            Polygon((LIP_X1 - LIP_LEADIN, -LIP_DEPTH), (LIP_X1 + 1.45, -LIP_DEPTH + 1.95),
                    (LIP_X1 + 1.45, -LIP_DEPTH - 0.6), (LIP_X1 - LIP_LEADIN, -LIP_DEPTH - 0.6))
        extrude(amount=PANEL_Y + 2.0, mode=Mode.SUBTRACT)

        # --- sensor hole (through the panel only) ---
        with BuildSketch(Plane.XY.offset(PANEL_T)):
            with Locations(SENSOR_HOLE_XY):
                Circle(SENSOR_HOLE_D / 2)
        extrude(amount=-(PANEL_T + 0.2), mode=Mode.SUBTRACT)

        # --- light slot over ALS + blue LED (stadium, through panel) ---
        _ax, _ay = LIGHT_SLOT_A
        _bx, _by = LIGHT_SLOT_B
        _sep = ((_bx - _ax) ** 2 + (_by - _ay) ** 2) ** 0.5
        _rot = degrees(atan2(_by - _ay, _bx - _ax))
        with BuildSketch(Plane.XY.offset(PANEL_T)):
            with Locations(((_ax + _bx) / 2, (_ay + _by) / 2)):
                SlotCenterToCenter(center_separation=_sep, height=LIGHT_SLOT_D,
                                   rotation=_rot)
        extrude(amount=-(PANEL_T + 0.2), mode=Mode.SUBTRACT)

        # --- orientation marking: engraved triangle pointing at local X=0 ---
        with BuildSketch(Plane.XY.offset(PANEL_T)):
            Polygon(MARK_APEX, MARK_BASE[0], MARK_BASE[1])
        extrude(amount=-MARK_DEPTH, mode=Mode.SUBTRACT)

        # --- chamfers: both ends of the sensor hole ---
        hole_edges = (fp.edges()
                      .filter_by(GeomType.CIRCLE)
                      .filter_by(lambda e: abs(e.radius - SENSOR_HOLE_D / 2) < 0.01))
        chamfer(hole_edges, HOLE_CHAMFER)

        # --- chamfers: light slot (wide on top for acceptance angle) ---
        def _slot_wire_edges(z):
            return (fp.edges()
                    .filter_by(lambda e: abs(e.bounding_box().min.Z - z) < 0.05
                               and abs(e.bounding_box().max.Z - z) < 0.05
                               and LIGHT_SLOT_A[0] - 4 < e.center().X < LIGHT_SLOT_B[0] + 4
                               and abs(e.center().Y - LIGHT_SLOT_A[1]) < 5))
        chamfer(_slot_wire_edges(PANEL_T), LIGHT_SLOT_TOP_CHAMFER)
        chamfer(_slot_wire_edges(0.0), HOLE_CHAMFER)

        # --- chamfer: top outer perimeter (elephant foot on the bed face) ---
        top_face = fp.faces().sort_by(Axis.Z)[-1]
        chamfer(top_face.outer_wire().edges(), TOP_CHAMFER)

    part = fp.part
    part.label = "faceplate"
    return part
