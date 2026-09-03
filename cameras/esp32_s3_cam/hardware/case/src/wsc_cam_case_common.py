"""Waveshare ESP32-S3-CAM-GC0308 case: shared parameters + part builders.

Units mm. Frame = the board's frame (ref/amz-esp32s3-cam-gc0308.py): origin PCB
plan bottom-left, Z=0 PCB bottom, camera head on +Z, USB-C on the Y=0 edge.

DEPLOYED POSE -- the change from the original sargineer design. The case stands
on its Y=0 wall, so in the field:

    board -Y  ->  DOWN     the underside USB-C and the lead slots open downward
    board +Y  ->  UP
    board +Z  ->  FORWARD  the lid is the front face; the camera looks out level

This is the easy board for the brief: its USB-C *and* its GH1.25 battery and
speaker connectors are already underside-mounted on the Y=0 edge, so standing
the case on that wall puts every connector on the floor with no rerouting. The
GC0308 head rides a ~60 mm FPC and is repositionable, so it mounts to the lid
(the front) independently of the PCB and the camera aims forward.

Two printed parts clamp the board with 4 x M2 screws running
lid -> lid post -> PCB hole -> body boss.

Weather: see cameras/hardware_common/caseskit.py. Splash resistant, not sealed.

PRINT: body floor-down (-Z face, the deployed BACK, on the bed -- the tripod pad
wants a brim). Lid outer-face-down. 0.4 nozzle, 0.2 layers, >=3 perimeters,
PETG or ASA -- not PLA, this lives outside.
"""
import sys, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent                # .../hardware/case/src
CASE = HERE.parent                                    # .../hardware/case
sys.path[:0] = [str(HERE), str(CASE / "ref"), str(CASE.parents[2] / "hardware_common")]
from pcbkit import *
from build123d import *
import caseskit as K

# --- deployed orientation ----------------------------------------------------------------
DOWN, UP = "-Y", "+Y"

# --- board (waveshare-esp32s3-cam-gc0308 part.yaml, sarg) -------------------------------
PCB_W, PCB_H, PCB_T = 37.0, 37.0, 1.6
PCB_R = 2.25
HOLES = [(2.2, 2.2), (34.8, 2.2), (2.2, 34.8), (34.8, 34.8)]   # dia 2.5 (est), M2
TOP_H = 7.5                                 # GC0308 head parked on the TF slot (est)
CAM_C = (20.0, 24.0)                        # camera head / TF slot block centre
USB_X, USB_W, USB_H, USB_OVER = 18.5, 9.0, 3.3, 1.3   # underside USB-C on the Y=0 edge
UNDER_H = 3.3                               # tallest underside item

# --- case parameters --------------------------------------------------------------------
CLR = 1.5                                   # PCB edge -> inner wall (covers the 1.3 overhang)
WALL = 2.4                                  # was 2.0; outdoor, and >=3 perimeters at 0.4
FLOOR_T = 2.4
UNDER_CLR = 5.0                             # PCB bottom -> inner floor (connectors + leads)
TOP_CLR = 0.9                               # camera head top -> lid underside
LID_T = 2.4
LIP_H, LIP_T, LIP_GAP = 2.5, 1.2, 0.2       # lip 1.5 -> 2.5: deeper labyrinth at the seam
POST_D = 5.0                                # lid posts / body bosses
BOSS_PILOT_D, BOSS_PILOT_DEPTH = 1.7, 6.0   # M2 self-tapping pilot
SCREW_THRU_D = 2.3
CBORE_D, CBORE_DEPTH = 4.2, 1.5
# Camera window: one conical cut, dia 18 inside flaring to dia 22 outside. The
# lens rides a repositionable FPC and its axis has never been measured, so the
# window stays generous; the flare is FOV relief and sheds water. Round rather
# than the old 18 square -- a cone is one operation, a square window needed a
# box plus a chamfer pass.
WINDOW, WINDOW_OUT = 18.0, 22.0
CORNER_R = 3.0
# weather -- angles, not ribs and grooves (see caseskit.face_chamfer / visor)
CHAMFER_PORT = 1.5
CHAMFER_ROOF = 1.5
VISOR_PROJ, VISOR_HEIGHT = 6.0, 6.0
# tripod: -Z face = the deployed BACK. Fully external (inner_h = 0) -- unlike the
# GOOUUU, this board's underside is a solid field of unmeasured GH1.25/FPC
# connectors, so nothing may intrude into the plenum. The cost is a 12.5 mm pad
# instead of 5; the alternative is trusting an estimated connector height with a
# heat-set insert directly beneath it.
TRIPOD = True
TRIPOD_XY = (18.5, 18.5)
TRIPOD_BOSS_D = 16.0
TRIPOD_PROUD, TRIPOD_INNER_H = 12.5, 0.0
TRIPOD_HOLE_D, TRIPOD_HOLE_DEPTH = 8.0, 13.5   # ruthex RX-1/4-20 (12.7 long)

# --- derived ----------------------------------------------------------------------------
X0, Y0 = -CLR - WALL, -CLR - WALL
OUT = PCB_W + 2 * (CLR + WALL)              # 44.8 square
IN = PCB_W + 2 * CLR                        # 40.0 cavity
Z_IN_FLOOR = -UNDER_CLR                     # -5.0
Z_BOT = Z_IN_FLOOR - FLOOR_T                # -7.4 body underside
Z_CEIL = PCB_T + TOP_H + TOP_CLR            # 10.0 body rim / lid underside
Z_TOP = Z_CEIL + LID_T
BODY_H = Z_CEIL - Z_BOT
CASE_COLOR = Color(0.25, 0.45, 0.75)
LO = (X0, Y0, Z_BOT)
HI_BODY = (X0 + OUT, Y0 + OUT, Z_CEIL)      # body only -- face features must not float
                                            # out over the lid's Z range
# --- floor openings (the Y=0 wall, once stood up) ----------------------------------------
USB_CUT_W = 13.0                            # plug-overmold sized
USB_CUT_Z0, USB_CUT_Z1 = -UNDER_CLR, 2.5   # overmold spans -4.9..1.6; 0.6 spare
USB_CUT_ZC = (USB_CUT_Z0 + USB_CUT_Z1) / 2
# One lead slot for the GH1.25 battery / speaker pigtails. Their exact XY is
# recorded as UNMEASURED, but the leads run inside the 5 mm plenum before they
# reach the wall, so they can arrive at a single slot from anywhere on the board
# -- the first pass had one either side, which was hedging against a problem the
# plenum already solves.
LEAD_W, LEAD_H = 10.0, 4.0
LEAD_X = 6.0
LEAD_ZC = Z_IN_FLOOR + LEAD_H / 2      # bottom edge on the floor line


def body():
    b = slab(OUT, OUT, BODY_H, r=CORNER_R, at=(X0, Y0, Z_BOT))
    b = b - box((-CLR, -CLR, Z_IN_FLOOR), (IN, IN, BODY_H))
    # corner bosses + gussets to the wall, pilot holes for the M2 screws
    for (x, y) in HOLES:
        b = b + cyl((x, y), Z_IN_FLOOR, POST_D, UNDER_CLR)
        gx0, gy0 = (-CLR, x) if x < PCB_W / 2 else (x, PCB_W + CLR)
        gy0_, gy1 = (-CLR, y) if y < PCB_H / 2 else (y, PCB_H + CLR)
        b = b + box((min(gx0, gy0), min(gy0_, gy1), Z_IN_FLOOR), (abs(gy0 - gx0), abs(gy1 - gy0_), UNDER_CLR))
        b = b - cyl((x, y), -BOSS_PILOT_DEPTH, BOSS_PILOT_D, BOSS_PILOT_DEPTH + 1)
    # Angles before openings: face_chamfer groups edges by axis, so the port and
    # roof faces must still be unbroken rectangles when it runs.
    b = K.face_chamfer(b, DOWN, CHAMFER_PORT)
    b = K.face_chamfer(b, UP, CHAMFER_ROOF)
    # tripod pad on the -Z face (deployed BACK)
    if TRIPOD:
        b = K.tripod_boss_z(b, TRIPOD_XY, Z_BOT, Z_IN_FLOOR,
                            proud=TRIPOD_PROUD, inner_h=TRIPOD_INNER_H,
                            boss_d=TRIPOD_BOSS_D, hole_d=TRIPOD_HOLE_D,
                            hole_depth=TRIPOD_HOLE_DEPTH)
    # --- the floor, in the deployed sense: the Y=0 wall ---
    # Both cuts already reach the cavity floor (z = Z_IN_FLOOR), so they drain
    # it; no separate drain hole is needed.
    b = K.port_slot(b, LO, HI_BODY, DOWN,
                    center_a=USB_X, w_a=USB_CUT_W,
                    center_b=USB_CUT_ZC, w_b=USB_CUT_Z1 - USB_CUT_Z0,
                    wall_t=WALL)
    b = K.port_slot(b, LO, HI_BODY, DOWN,
                    center_a=LEAD_X, w_a=LEAD_W,
                    center_b=LEAD_ZC, w_b=LEAD_H,
                    wall_t=WALL, lead_in=0.4)
    b.label, b.color = "wsc_cam_case_body", CASE_COLOR
    return b


def lid():
    l = slab(OUT, OUT, LID_T, r=CORNER_R, at=(X0, Y0, Z_CEIL))
    lip_o = IN - 2 * LIP_GAP
    l = l + (box((-CLR + LIP_GAP, -CLR + LIP_GAP, Z_CEIL - LIP_H), (lip_o, lip_o, LIP_H))
             - box((-CLR + LIP_GAP + LIP_T, -CLR + LIP_GAP + LIP_T, Z_CEIL - LIP_H - 1),
                   (lip_o - 2 * LIP_T, lip_o - 2 * LIP_T, LIP_H + 2)))
    for (x, y) in HOLES:
        l = l + cyl((x, y), PCB_T, POST_D, Z_CEIL - PCB_T)
    for (x, y) in HOLES:
        l = l - cyl((x, y), PCB_T - 1, SCREW_THRU_D, Z_TOP - PCB_T + 2)
        l = l - cyl((x, y), Z_TOP - CBORE_DEPTH, CBORE_D, CBORE_DEPTH + 1)
    l = K.conical_window(l, CAM_C, Z_TOP, Z_CEIL, WINDOW, WINDOW_OUT)
    # visor over the window: "above" is +Y in the deployed pose
    l = K.visor(l, Z_TOP, UP, center_across=CAM_C[0],
                width=WINDOW_OUT + 2 * VISOR_PROJ,
                above=CAM_C[1] + WINDOW_OUT / 2 + 1.5,
                proj=VISOR_PROJ, height=VISOR_HEIGHT)
    l.label, l.color = "wsc_cam_case_lid", CASE_COLOR
    return l


_BOARD_NAME = "amz-esp32s3-cam-gc0308.py"


def board_ref():
    path = CASE / "ref" / _BOARD_NAME
    spec = importlib.util.spec_from_file_location("wsc_board", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    b = m.gen_step()
    b.label, b.color = "waveshare_esp32s3_cam_gc0308_board", PCB_COLOR
    return b
