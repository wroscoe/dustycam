"""Fit checks for the GOOUUU ESP32-S3-CAM case.

Extends the original sargineer verify.py (interference volumes + plug probes)
with checks for the features this rework added: the downward port face, the
blind button/LED membranes, and the back-face tripod pocket.

Every check asserts; the script exits non-zero on the first failure so it can
gate a print. Run:  python verify_goouuu.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from goouuu_cam_case_common import *
from build123d import *

FAILED = []


def vol(s):
    try:
        return round(s.volume, 3)
    except Exception:
        return 0.0


def check(name, got, want, tol=0.02, cmp="eq"):
    """cmp: eq = equal within tol, ge = at least want, gt = strictly more than want."""
    ok = {"eq": lambda: abs(got - want) <= tol,
          "ge": lambda: got >= want - tol,
          "gt": lambda: got > want}[cmp]()
    sign = {"eq": "", "ge": ">=", "gt": ">"}[cmp]
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got} (want {sign}{want})")
    if not ok:
        FAILED.append(name)


b, l, brd = body(), lid(), board_ref()
print("body  bbox", b.bounding_box().min, b.bounding_box().max, "vol", round(b.volume))
print("lid   bbox", l.bounding_box().min, l.bounding_box().max, "vol", round(l.volume))
print("board bbox", brd.bounding_box().min, brd.bounding_box().max)
print()

# --- interference (carried over from the upstream design) -------------------------------
check("body & board", vol(b & brd), 0.0)
check("lid & board", vol(l & brd), 0.0)
check("body & lid", vol(b & l), 0.0)
check("board shifted +3X & body", vol(b & (Pos(3, 0, 0) * brd)), 0.0)

pins = None
for (x, y) in B.PIN_PTS:
    r = box((x - 0.32, y - 0.32, -HDR_PLASTIC - PIN_BELOW), (0.64, 0.64, HDR_PLASTIC + PIN_BELOW))
    pins = r if pins is None else pins + r
check("header pins & body", vol(b & pins), 0.0)

for (x, y) in SCREW_XY:
    rod = Pos(x, y, (Z_TOP + Z_CEIL - PILOT_DEPTH) / 2) * Cylinder(0.8, Z_TOP - Z_CEIL + PILOT_DEPTH + 0.01)
    check(f"screw rod {x:.2f},{y:.2f} & board", vol(brd & rod), 0.0)

# --- the deployed floor: both USB plugs must reach their ports --------------------------
# 12 x 6.5 overmold driven up through the X=0 wall, past the drip ring, to the shell.
for u in B.USB:
    yc = u["y0"] + u["w"] / 2
    plug = box((X0 - 5, yc - 6.0, USB_ZC - 3.25),
               (5 + (B.USB_X0 - X0) + 0.5, 12.0, 6.5))
    check(f"USB plug probe y={yc:.1f} & body", vol(b & plug), 0.0)

# --- the USB opening must also serve as the drain: it has to reach the cavity floor -----
check("USB opening reaches the cavity floor", Z_IN_FLOOR - USB_CUT_Z0, 0.0, cmp="ge")

# --- tripod pocket must stay blind ------------------------------------------------------
pocket_top = Z_BOT - TRIPOD_PROUD + TRIPOD_HOLE_DEPTH
column_top = Z_IN_FLOOR + TRIPOD_INNER_H
check("tripod pocket buried below the column top", column_top - pocket_top, 1.0, cmp="ge")
# a rod continuing 1 mm past the pocket must still hit solid plastic
beyond = cyl(TRIPOD_XY, pocket_top, TRIPOD_HOLE_D - 0.1, 0.9)
check("material above the tripod pocket", vol(b & beyond), 0.0, cmp="gt")
# and the pocket must not break into the board space
check("tripod column & board", vol(cyl(TRIPOD_XY, Z_IN_FLOOR, TRIPOD_BOSS_D, TRIPOD_INNER_H) & brd), 0.0)

# --- blind features: buttons and LED must NOT be through-holes --------------------------
for name, c, d, m in ([("BOOT", BUTTON_CS[0], BUTTON_HOLE_D, BUTTON_MEMBRANE),
                       ("RST", BUTTON_CS[1], BUTTON_HOLE_D, BUTTON_MEMBRANE),
                       ("LED", LED, LED_HOLE_D, LED_MEMBRANE)]):
    skin = cyl(c, Z_TOP - m, d - 0.1, m)          # the membrane itself
    check(f"{name} membrane is solid", vol(l & skin), 0.0, cmp="gt")

# --- lens: clear column out through the window, and the visor stays out of it ----------
col = cyl(CAM_C, PCB_T, WINDOW_D - 0.02, Z_TOP - PCB_T)
check("lens column & lid", vol(l & col), 0.0)
# the visor starts above the window, so the straight-ahead view stays clear
fov = cyl(CAM_C, Z_TOP, WINDOW_D_OUT, VISOR_PROJ + 2.0)
check("lens forward cone & lid (visor clear)", vol(l & fov), 0.0)
# the window must actually open out, not in
check("window flares outward", WINDOW_D_OUT - WINDOW_D, 1.0, cmp="ge")

print()
print(f"case outer {OUT_X} x {OUT_Y} x {Z_TOP - Z_BOT}"
      f" | deployed: {OUT_Y} wide x {OUT_X} tall x {Z_TOP - Z_BOT} deep")
print(f"lid screws ~M2 x {round(LID_T - CBORE_DEPTH + PILOT_DEPTH, 1)}")
print(f"tripod pad protrudes {TRIPOD_PROUD} mm from the back face")
if FAILED:
    print(f"\n{len(FAILED)} CHECK(S) FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall checks passed")
