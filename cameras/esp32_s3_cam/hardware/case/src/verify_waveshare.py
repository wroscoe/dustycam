"""Fit checks for the Waveshare ESP32-S3-CAM-GC0308 case.

Extends the original sargineer verify.py with checks for what this rework
added: the downward port face, the lead slots, the drain, and the fully
external tripod pocket. Every check asserts; exits non-zero on failure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsc_cam_case_common import *
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

# --- interference -----------------------------------------------------------------------
# NOTE: this was 8.431 mm3 in the published design -- two corner bosses lapped the
# underside connector envelope. Fixed at source in ref/amz-esp32s3-cam-gc0308.py.
check("body & board", vol(b & brd), 0.0)
check("lid & board", vol(l & brd), 0.0)
check("body & lid", vol(b & l), 0.0)

for (x, y) in HOLES:
    rod = Pos(x, y, (Z_TOP + PCB_T) / 2) * Cylinder(0.8, Z_TOP - PCB_T)
    check(f"screw rod {x},{y} & board", vol(brd & rod), 0.0)

# --- the deployed floor -----------------------------------------------------------------
# USB-C plug overmold (12 x 6.5) driven up through the Y=0 wall; the overmold is
# centred on the shell, which hangs -USB_H..0 under the PCB
plug = box((USB_X - 6.0, Y0 - 5, -USB_H / 2 - 3.25),
           (12.0, 5 + (CLR + WALL) + 0.5, 6.5))
check("USB plug probe & body", vol(b & plug), 0.0)

lead = box((LEAD_X - LEAD_W / 2 + 0.5, Y0 - 1, LEAD_ZC - LEAD_H / 2 + 0.5),
           (LEAD_W - 1.0, WALL + 2, LEAD_H - 1.0))
check(f"lead slot x={LEAD_X} & body", vol(b & lead), 0.0)

# with no separate drain, both floor cuts must reach the cavity floor line
check("USB opening reaches the floor", Z_IN_FLOOR - USB_CUT_Z0, 0.0, cmp="ge")
check("lead slot reaches the floor", Z_IN_FLOOR - (LEAD_ZC - LEAD_H / 2), 0.0, cmp="ge")

# --- tripod pocket must stay blind and stay out of the plenum ---------------------------
pocket_top = Z_BOT - TRIPOD_PROUD + TRIPOD_HOLE_DEPTH
check("tripod pocket stops below the inner floor", Z_IN_FLOOR - pocket_top, 1.0, cmp="ge")
beyond = cyl(TRIPOD_XY, pocket_top, TRIPOD_HOLE_D - 0.1, 0.9)
check("material above the tripod pocket", vol(b & beyond), 0.0, cmp="gt")
plenum = box((-CLR, -CLR, Z_IN_FLOOR), (IN, IN, UNDER_CLR))
check("tripod boss & plenum", vol(cyl(TRIPOD_XY, Z_BOT - TRIPOD_PROUD, TRIPOD_BOSS_D,
                                      TRIPOD_PROUD + FLOOR_T) & plenum), 0.0)

# --- lens window and visor --------------------------------------------------------------
col = cyl(CAM_C, PCB_T, WINDOW - 0.02, Z_TOP - PCB_T)
check("camera column & lid", vol(l & col), 0.0)
fov = cyl(CAM_C, Z_TOP, WINDOW_OUT, VISOR_PROJ + 2.0)
check("forward cone & lid (visor clear)", vol(l & fov), 0.0)
check("window flares outward", WINDOW_OUT - WINDOW, 1.0, cmp="ge")

print()
print(f"case outer {OUT} x {OUT} x {Z_TOP - Z_BOT}"
      f" | deployed: {OUT} wide x {OUT} tall x {Z_TOP - Z_BOT} deep")
print(f"board screws ~M2 x {round(Z_TOP - PCB_T + BOSS_PILOT_DEPTH - CBORE_DEPTH, 1)}")
print(f"tripod pad protrudes {TRIPOD_PROUD} mm from the back face")
if FAILED:
    print(f"\n{len(FAILED)} CHECK(S) FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall checks passed")
