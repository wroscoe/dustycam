"""GOOUUU ESP32-S3-CAM case: shared parameters + part builders (body + lid).

Units mm. Frame = the board's frame (ref/goouuu_board_ref.py): origin PCB plan
bottom-left, +X along the long edge, USB-C ports on the X=0 end, Z=0 PCB
bottom, lens looks +Z.

DEPLOYED POSE -- this is the change from the original sargineer design.
The case stands on its X=0 end wall, so in the field:

    board -X  ->  DOWN     both USB-C ports open through the floor
    board +X  ->  UP       the WROOM antenna overhang sits at the top
    board +Z  ->  FORWARD  the lid is the front face; the lens looks out level

The original laid the case flat with a tripod boss on the -Y wall, which its
own notes flagged: "the camera then looks sideways". Standing it on the
connector end fixes the aim and puts the ports underneath in one move, which
is what an outdoor camera wants anyway -- a downward opening cannot pool water
or catch falling rain.

Retention scheme (no screws through the board - its holes are photo-located only):
  * the two 2x20 pin-header rows (25.4 mm apart, pins pointing down) drop into
    two grooves in bed rails on the body floor -> exact Y location, header
    plastic seats on the rails -> Z location (PCB bottom at Z=0);
  * the body's USB-end corner blocks stop the PCB's X=0 edge and carry ledges
    under the PCB corners; the lid's corner feet clamp those corners (tapered
    pins probe the board's corner holes; snip them if they miss);
  * a lid pad rests 0.3 mm over the WROOM shield can at the antenna end;
  * lid -> body: 4 x M2 x 8 self-tapping screws into full-height corner bosses.

Weather: see cameras/hardware_common/caseskit.py for the strategy. Splash
resistant, not sealed.

PRINT: body floor-down (the -Z face, i.e. the deployed BACK, on the bed; the
tripod pad then needs a brim, no supports). Lid outer-face-down. 0.4 nozzle,
0.2 layers, >=3 perimeters, PETG or ASA -- not PLA, this lives outside.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent                # .../hardware/case/src
CASE = HERE.parent                                    # .../hardware/case
sys.path[:0] = [str(HERE), str(CASE / "ref"), str(CASE.parents[2] / "hardware_common")]
from pcbkit import *
from build123d import *
import caseskit as K
import goouuu_board_ref as B

# --- deployed orientation ----------------------------------------------------------------
DOWN, UP = "-X", "+X"                        # which board axis points at the ground

# --- board facts used by the case (see ref/goouuu_board_ref.py for provenance) -----------
PCB_L, PCB_W, PCB_T = B.PCB_L, B.PCB_W, B.PCB_T
ROW_Y, PIN_X0, PITCH, PIN_N = B.ROW_Y, B.PIN_X0, B.PITCH, B.PIN_N
HDR_PLASTIC, PIN_BELOW = B.HDR_PLASTIC, B.PIN_BELOW
HOLES = B.HOLES
TOP_H = B.TOP_H                              # 8.5: lens top above PCB top (est)
CAN_TOP = PCB_T + B.MODULE["h"]              # 4.7
ANT_END = B.ANT_END                          # 68.0 nominal; cavity sized for 69.3
CAM_C = B.CAM_C
BUTTON_CS = [(x + w / 2, y + d / 2) for (x, y, w, d, h) in B.BUTTONS]
LED = B.LED
USB_Y0, USB_Y1 = B.USB[0]["y0"], B.USB[1]["y0"] + B.USB[1]["w"]   # 4.2 .. 24.4 shells
USB_ZC = PCB_T + B.USB_H / 2                 # 3.25 port centre

# --- case parameters --------------------------------------------------------------------
WALL = 2.4                                   # was 2.0; outdoor, and >=3 perimeters at 0.4
FLOOR_T = 2.4
LID_T = 2.4
CLR_Y = 2.0                                  # PCB long edge -> inner wall
CLR_X0 = 4.5                                 # USB end: corner screw blocks live here
X_ANT_MAX = PCB_L + 6.3                      # 69.3 worst-case antenna tip
CLR_X1 = 1.2
TOP_CLR = 0.9                                # lens top -> lid underside
CORNER_R = 3.0
# bed rails + pin grooves
RAIL_TOP = -HDR_PLASTIC                      # -2.5 header plastic seats here
GROOVE_W, GROOVE_CH = 1.5, 0.3
GROOVE_DEPTH = PIN_BELOW + 0.5               # 6.5
RAIL_IN = 4.0
RAIL_X0, RAIL_X1 = 8.5, 63.5
# corner screw bosses (full height), M2 self-tap
BOSS_D, PILOT_D, PILOT_DEPTH = 5.5, 1.7, 8.0
SCREW_THRU_D, CBORE_D, CBORE_DEPTH = 2.3, 4.2, 1.5
BLOCK_Y = 2.6
LEDGE_X1, LEDGE_GAP = 3.0, 0.1
STOP_X = -0.3
# lid
LIP_H, LIP_T, LIP_GAP = 2.5, 1.2, 0.2        # lip 1.5 -> 2.5: deeper labyrinth at the seam
FOOT = (0.5, 0.3, 4.0, 3.4)
PAD = (53.5, 10.0, 6.0, 9.0)
PAD_GAP = 0.3
# lens window: one conical cut, dia 12 at the inside opening flaring to dia 16
# outside. Lens is dia 7 at CAM_C +/-1, so 12.0 leaves >=2 mm of slop all round
# and the flare gives the FOV relief the old square rebate needed a second cut for.
WINDOW_D, WINDOW_D_OUT = 12.0, 16.0
# buttons/LED are blind: nothing here has to pass a plug, so nothing here is a hole
BUTTON_HOLE_D, BUTTON_MEMBRANE = 6.0, 0.6    # 3 layers of PETG still presses
LED_HOLE_D, LED_MEMBRANE = 3.0, 0.8
# weather -- angles, not ribs and grooves (see caseskit.face_chamfer / visor)
CHAMFER_PORT = 1.5                           # breaks the arris on the floor face
CHAMFER_ROOF = 1.5                           # ditto the roof
VISOR_PROJ, VISOR_HEIGHT = 6.0, 6.0
SD_SLOT = False                              # the +X wall is the ROOF when stood up;
                                             # an open slot there would drink. Card
                                             # access = take the lid off.
# tripod: on the -Z face, which is the deployed BACK. Split between an external
# pad and an internal column so the bump stays 5 mm instead of 12.5.
TRIPOD = True
TRIPOD_XY = (33.0, 14.5)                     # centred; clear of the pin rows and the TF slot
TRIPOD_BOSS_D = 16.0
TRIPOD_PROUD, TRIPOD_INNER_H = 5.0, 8.0
TRIPOD_HOLE_D, TRIPOD_HOLE_DEPTH = 8.0, 13.5   # ruthex RX-1/4-20 (12.7 long)

# --- derived ----------------------------------------------------------------------------
XI0, XI1 = -CLR_X0, X_ANT_MAX + CLR_X1       # cavity x -4.5 .. 70.5
YI0, YI1 = -CLR_Y, PCB_W + CLR_Y             # cavity y -2 .. 31
X0, Y0 = XI0 - WALL, YI0 - WALL
OUT_X, OUT_Y = (XI1 - XI0) + 2 * WALL, (YI1 - YI0) + 2 * WALL
Z_GROOVE = RAIL_TOP - GROOVE_DEPTH           # -9.0
Z_IN_FLOOR = Z_GROOVE + 0.5                  # -8.5
Z_BOT = Z_IN_FLOOR - FLOOR_T
Z_CEIL = PCB_T + TOP_H + TOP_CLR             # 11.0 wall top / lid underside
Z_TOP = Z_CEIL + LID_T
BODY_H = Z_CEIL - Z_BOT
BOSS_XY = [(XI1 - BOSS_D / 2, YI0 + BOSS_D / 2), (XI1 - BOSS_D / 2, YI1 - BOSS_D / 2)]
BLOCK_PILOT_XY = [(XI0 / 2, (YI0 + BLOCK_Y) / 2), (XI0 / 2, (YI1 + PCB_W - BLOCK_Y) / 2)]
SCREW_XY = BOSS_XY + BLOCK_PILOT_XY
CASE_COLOR = Color(0.25, 0.45, 0.75)
LO = (X0, Y0, Z_BOT)                         # case bounding corners, for caseskit
HI = (X0 + OUT_X, Y0 + OUT_Y, Z_TOP)         # whole case (body + lid)
HI_BODY = (X0 + OUT_X, Y0 + OUT_Y, Z_CEIL)   # body only -- face features must not
                                             # float out over the lid's Z range
# USB opening: both plug overmolds (12 x 6.5) through the now-downward X=0 wall
USB_CUT_Y0, USB_CUT_Y1 = BLOCK_Y, PCB_W - BLOCK_Y      # 2.6 .. 26.4
# Z0 runs all the way down to the cavity floor. Standing on this wall, z is a
# HORIZONTAL axis, so water on the inner floor has no reason to flow toward a
# z-limited opening -- it just sits behind it. Taking the opening to the floor
# line makes it the drain too, which is why there is no separate drain hole.
USB_CUT_Z0, USB_CUT_Z1 = Z_IN_FLOOR, 7.0
USB_CUT_YC = (USB_CUT_Y0 + USB_CUT_Y1) / 2
USB_CUT_ZC = (USB_CUT_Z0 + USB_CUT_Z1) / 2


def _groove(y):
    g = box((RAIL_X0 - 1, y - GROOVE_W / 2, Z_GROOVE), (RAIL_X1 - RAIL_X0 + 2, GROOVE_W, GROOVE_DEPTH + 1))
    g = g + box((RAIL_X0 - 1, y - GROOVE_W / 2 - GROOVE_CH, RAIL_TOP - GROOVE_CH), (RAIL_X1 - RAIL_X0 + 2, GROOVE_W + 2 * GROOVE_CH, GROOVE_CH + 1))
    return g


def body():
    b = slab(OUT_X, OUT_Y, BODY_H, r=CORNER_R, at=(X0, Y0, Z_BOT))
    b = b - box((XI0, YI0, Z_IN_FLOOR), (XI1 - XI0, YI1 - YI0, BODY_H))
    # bed rails with pin grooves along both long walls
    b = b + box((RAIL_X0, YI0, Z_IN_FLOOR), (RAIL_X1 - RAIL_X0, RAIL_IN - YI0, RAIL_TOP - Z_IN_FLOOR))
    b = b + box((RAIL_X0, PCB_W - RAIL_IN, Z_IN_FLOOR), (RAIL_X1 - RAIL_X0, YI1 - (PCB_W - RAIL_IN), RAIL_TOP - Z_IN_FLOOR))
    for y in ROW_Y:
        b = b - _groove(y)
    # USB-end corner blocks (screw bosses + X stop) with ledges under the PCB corners
    for (y0, y1) in ((YI0, BLOCK_Y), (PCB_W - BLOCK_Y, YI1)):
        b = b + box((XI0, y0, Z_IN_FLOOR), (STOP_X - XI0, y1 - y0, Z_CEIL - Z_IN_FLOOR))
        b = b + box((STOP_X, y0, Z_IN_FLOOR), (LEDGE_X1 - STOP_X, y1 - y0, -LEDGE_GAP - Z_IN_FLOOR))
    # +X corner bosses, full height
    for (x, y) in BOSS_XY:
        b = b + cyl((x, y), Z_IN_FLOOR, BOSS_D, Z_CEIL - Z_IN_FLOOR)
    for (x, y) in SCREW_XY:
        b = b - cyl((x, y), Z_CEIL - PILOT_DEPTH, PILOT_D, PILOT_DEPTH + 1)
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
    # --- the floor, in the deployed sense: the X=0 wall ---
    # The USB opening is the only cut here. It reaches the cavity floor, so it
    # is also the drain -- the separate drain hole the first pass had was doing
    # nothing this one does not.
    b = K.port_slot(b, LO, HI_BODY, DOWN,
                    center_a=USB_CUT_YC, w_a=USB_CUT_Y1 - USB_CUT_Y0,
                    center_b=USB_CUT_ZC, w_b=USB_CUT_Z1 - USB_CUT_Z0,
                    wall_t=WALL)
    if SD_SLOT:
        b = b - box((XI1 - 1, 7.0, -4.0), (WALL + 2, 15.0, 4.0))
    b.label, b.color = "goouuu_cam_case_body", CASE_COLOR
    return b


def lid():
    l = slab(OUT_X, OUT_Y, LID_T, r=CORNER_R, at=(X0, Y0, Z_CEIL))
    lip = box((XI0 + LIP_GAP, YI0 + LIP_GAP, Z_CEIL - LIP_H), (XI1 - XI0 - 2 * LIP_GAP, YI1 - YI0 - 2 * LIP_GAP, LIP_H)) \
        - box((XI0 + LIP_GAP + LIP_T, YI0 + LIP_GAP + LIP_T, Z_CEIL - LIP_H - 1), (XI1 - XI0 - 2 * (LIP_GAP + LIP_T), YI1 - YI0 - 2 * (LIP_GAP + LIP_T), LIP_H + 2))
    for (x, y) in BOSS_XY:
        lip = lip - cyl((x, y), Z_CEIL - LIP_H - 1, BOSS_D + 2 * LIP_GAP, LIP_H + 2)
    for (y0, y1) in ((YI0 - 1, BLOCK_Y + LIP_GAP), (PCB_W - BLOCK_Y - LIP_GAP, YI1 + 1)):
        lip = lip - box((XI0 - 1, y0, Z_CEIL - LIP_H - 1), (STOP_X - XI0 + 1 + LIP_GAP, y1 - y0, LIP_H + 2))
    # the lip must also clear the USB opening's mouth flare at the X=0 wall
    l = l + lip
    # corner feet clamping the PCB corners.
    # The upstream design also grew tapered pins probing the board's dia 2.4
    # corner holes, with the note "snip them if they miss" -- those holes are
    # photo-located, and the header grooves already fix the board in X and Y.
    # A locating feature you are told to cut off is not one; dropped.
    fx, fy, fw, fd = FOOT
    for y0 in (fy, PCB_W - fy - fd):
        l = l + box((fx, y0, PCB_T), (fw, fd, Z_CEIL - PCB_T))
    # pad over the WROOM shield can
    px, py, pw, pd = PAD
    l = l + box((px, py, CAN_TOP + PAD_GAP), (pw, pd, Z_CEIL - CAN_TOP - PAD_GAP))
    # screw holes + counterbores
    for (x, y) in SCREW_XY:
        l = l - cyl((x, y), Z_CEIL - LIP_H - 1, SCREW_THRU_D, LID_T + LIP_H + 2)
        l = l - cyl((x, y), Z_TOP - CBORE_DEPTH, CBORE_D, CBORE_DEPTH + 1)
    # lens window: one conical cut, flaring outward
    l = K.conical_window(l, CAM_C, Z_TOP, Z_CEIL, WINDOW_D, WINDOW_D_OUT)
    # visor over the window: "above" is +X in the deployed pose. Nothing hangs
    # below where the wedge starts, so it only has to clear the window itself.
    l = K.visor(l, Z_TOP, UP, center_across=CAM_C[1],
                width=WINDOW_D_OUT + 2 * VISOR_PROJ,
                above=CAM_C[0] + WINDOW_D_OUT / 2 + 1.5,
                proj=VISOR_PROJ, height=VISOR_HEIGHT)
    # buttons and LED stay blind -- nothing here passes a plug, so nothing is a hole
    for c in BUTTON_CS:
        l = K.membrane_hole(l, c, BUTTON_HOLE_D, Z_TOP, Z_CEIL, membrane=BUTTON_MEMBRANE)
    l = K.membrane_hole(l, LED, LED_HOLE_D, Z_TOP, Z_CEIL, membrane=LED_MEMBRANE)
    l.label, l.color = "goouuu_cam_case_lid", CASE_COLOR
    return l


def board_ref():
    b = B.gen_step()
    b.label = "goouuu_esp32s3cam_board"
    return b
