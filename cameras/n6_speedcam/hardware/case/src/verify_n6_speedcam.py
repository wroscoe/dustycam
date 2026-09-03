"""Fit checks for the n6_speedcam case.

Boolean interference between each printed part and every reference part
(N6, radar, DFR0535, carrier, battery envelope), plug/gland/drain probes
through the floor, the blind tripod pocket, the radome thickness and the
lens column. Every check asserts; the script exits non-zero on the first
failure so it can gate a print.  Run:

    ~/.claude/skills/cad/.venv/bin/python verify_n6_speedcam.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from n6_speedcam_case_common import *
from build123d import *

FAILED = []


def vol(s):
    try:
        return round(s.volume, 3)
    except Exception:
        return 0.0


def check(name, got, want, tol=0.02, cmp="eq"):
    """cmp: eq = equal within tol, ge = at least want, gt = strictly more than want, le = at most."""
    ok = {"eq": lambda: abs(got - want) <= tol,
          "ge": lambda: got >= want - tol,
          "gt": lambda: got > want,
          "le": lambda: got <= want + tol}[cmp]()
    sign = {"eq": "", "ge": ">=", "gt": ">", "le": "<="}[cmp]
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got} (want {sign}{want})")
    if not ok:
        FAILED.append(name)


def solids_of(compound):
    return list(compound.solids())


b, l = body(), lid()
parts = {p.label: p for p in refs()}
print("body bbox", b.bounding_box().min, b.bounding_box().max, "vol", round(b.volume))
print("lid  bbox", l.bounding_box().min, l.bounding_box().max, "vol", round(l.volume))
for name, p in parts.items():
    bb = p.bounding_box()
    print(f"{name:22s} bbox", bb.min, bb.max)

check("body is one solid", len(b.solids()), 1)
check("lid is one solid", len(l.solids()), 1)

# --- interference: every case part against every reference solid ---------------------------
for case_name, case in (("body", b), ("lid", l)):
    for name, p in parts.items():
        total = 0.0
        for s in solids_of(p):
            total += vol(case & s)
        check(f"{case_name} vs {name} interference mm3", total, 0.0)
core, ribs = lid_core(), crush_ribs()
check("lid (without ribs) vs body interference mm3", vol(core & b), 0.0)
check("lid vs body interference == crush ribs only (designed)", vol(l & b), vol(ribs & b), tol=0.05)
check("crush ribs bite (designed interference > 5 mm3)", vol(ribs & b), 5.0, cmp="gt")

# --- reference parts against each other (the whole stack must nest) ---------------------------
names = list(parts)
for i, a in enumerate(names):
    for c in names[i + 1:]:
        total = 0.0
        for sa in solids_of(parts[a]):
            for sc in solids_of(parts[c]):
                total += vol(sa & sc)
        check(f"{a} vs {c} interference mm3", total, 0.0)

# --- the floor: USB plug overmold passes, drain reaches the floor line, gland is clear ---------
plug = box((USB_XC - 6.0, Y0 - 8.0, USB_ZC - 3.25), (12.0, 8.0 + WALL + 6.0, 6.5))
check("USB-C plug overmold (12 x 6.5) passes the floor opening", vol(plug & b), 0.0)
drain = box((DRAIN_XC - DRAIN_W / 2 + 0.3, Y0 - 1.0, Z_IN_FLOOR), (DRAIN_W - 0.6, WALL + 2.0, DRAIN_H - 0.3))
check("drain opening reaches the cavity floor line", vol(drain & b), 0.0)
gx, gz = GLAND_XZ
gland = Pos(gx, Y0 - 1.0, gz) * Rot(-90, 0, 0) * Cylinder((GLAND_D - 0.2) / 2, WALL + 2.0,
                                                          align=(Align.CENTER, Align.CENTER, Align.MIN))
check("PG7 gland bore (dia 12.3) is clear", vol(gland & b), 0.0)
nut = Pos(gx, YI0, gz) * Rot(-90, 0, 0) * Cylinder(18.0 / 2, 8.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
for name in ("dfr0535", "hlk_ld2415h", "carrier"):
    total = sum(vol(nut & s) for s in solids_of(parts[name]))
    check(f"gland nut (dia 18 x 8 inside) clears {name}", total, 0.0)
check("gland nut clears the body walls/bosses", vol(nut & b), 0.0)

# --- tripod: blind pocket, insert fits, plastic remains above it -------------------------------
pocket = cyl(TRIPOD_XY, Z_BOT - 0.5, TRIPOD_HOLE_D - 0.1, TRIPOD_HOLE_DEPTH + 0.5)
check("tripod pocket takes the dia 8 x 13.5 insert", vol(pocket & b), 0.0)
plug = cyl(TRIPOD_XY, Z_BOT + TRIPOD_HOLE_DEPTH, TRIPOD_HOLE_D, 1.0)
check("tripod pocket is blind (>=1.0 mm plastic above it)", vol(plug & b), vol(plug))
bat = parts["lipo_pouch_envelope"]
column = cyl(TRIPOD_XY, Z_IN_FLOOR, TRIPOD_BOSS_D, TRIPOD_INNER_H)
check("tripod column stays clear of the battery envelope", vol(column & bat), 0.0)
check("back face is flat (nothing below Z_BOT)", b.bounding_box().min.Z, Z_BOT)

# --- radome + lens window ---------------------------------------------------------------------
px, py, pw, ph = POCKET
probe = box((px + 0.5, py + 0.5, Z_CEIL - 0.5), (pw - 1.0, ph - 1.0, LID_T + 1.0))
inner = box((px + 0.5, py + 0.5, Z_RADOME_IN - 0.001), (pw - 1.0, ph - 1.0, RADOME_T + 0.002))
# everything the probe finds inside the pocket footprint must be the radome membrane plus the 4 bosses
bosses = sum(vol(cyl(h, Z_RADAR_FACE, RADAR_BOSS_D, RADAR_STANDOFF) & probe & l) for h in RADAR_HOLES)
check("radome: pocket region holds only the membrane + 4 bosses",
      vol(probe & l), vol(inner & l) + bosses, tol=1.0)
check("radome membrane is RADOME_T thick", vol(inner & l) / ((pw - 1.0) * (ph - 1.0)), RADOME_T, tol=0.01)
lens = cyl(CAM_C, Z_CEIL - 1.0, WINDOW_D - 0.1, LID_T + 2.0)
check("lens column (dia 14.9) is clear through the lid", vol(lens & l), 0.0)
flare = cyl(CAM_C, Z_TOP - 0.3, WINDOW_D_OUT - 0.6, 0.3)
check("window flares outward (dia 18.4 clear at the outer face)", vol(flare & l), 0.0)

# --- bosses actually reach their parts (gap, not overlap: the interference checks cover overlap)
n6 = parts["openmv_n6"]
for (x, y) in N6_MOUNT:
    rod = cyl((x, y), N6_PCB_T - 3.0, 2.4, 3.0 + N6_PILOT_DEPTH + 1.0)   # M2.5 from behind
    check(f"N6 screw rod at ({x:.2f},{y:.2f}) passes only the mounting hole", vol(rod & b), 0.0)
    check(f"N6 boss lands on the PCB at ({x:.2f},{y:.2f})",
          vol(cyl((x, y), N6_PCB_T, N6_TIP_D, 0.5) & l), vol(cyl((x, y), N6_PCB_T, N6_TIP_D, 0.5)) - vol(cyl((x, y), N6_PCB_T, N6_PILOT_D, 0.5)), tol=0.5)
for (x, y) in RADAR_HOLES:
    rod = cyl((x, y), Z_RADAR_FACE - RD.PCB_T - 4.0, 2.4, RD.PCB_T + 4.0 + RADAR_STANDOFF)
    check(f"radar screw rod at ({x:.2f},{y:.2f}) clears the radar's own parts",
          sum(vol(rod & s) for s in solids_of(parts["hlk_ld2415h"]) if s.label != "ld2415h_pcb"), 0.0)
for (x, y) in DFR_HOLES:
    rod = cyl((x, y), Z_DFR_PCB, 2.8, PM.PCB_T + 6.0)
    check(f"DFR0535 screw rod at ({x:.1f},{y:.1f}) clears its parts",
          sum(vol(rod & s) for s in solids_of(parts["dfr0535"]) if s.label != "dfr0535_pcb"), 0.0)

# --- clearances that are tight by design -----------------------------------------------------
check("carrier (plugs included) sits wholly above the N6 (y gap)",
      parts["carrier"].bounding_box().min.Y - parts["openmv_n6"].bounding_box().max.Y, 1.0, cmp="ge")

# --- the hinged front: top edge in the pocket, rotate closed, captured front and back --------
PIVOT = Axis((0, LID_Y1, Z_CEIL), (1, 0, 0))          # the lid's top-back edge
def tilted(shape, deg):
    return shape.rotate(PIVOT, -deg)                   # bottom edge swings out (+Z)
for deg in (2.0, 4.0, 6.0):
    check(f"door rotates in: lid at {deg:.0f} deg clears the body", vol(tilted(core, deg) & b), 0.0)
    swung = [tilted(p, deg) for name, p in parts.items() if name in ("openmv_n6", "hlk_ld2415h", "carrier")]
    hit = sum(vol(sld & b) for p in swung for sld in solids_of(p))
    hit += sum(vol(sld & d) for p in swung for sld in solids_of(p) for d in solids_of(parts["dfr0535"]))
    check(f"door rotates in: lid-mounted parts at {deg:.0f} deg clear body + DFR0535", hit, 0.0)
check("pocket lip holds the top: lid pushed +1 forward hits it", vol((Pos(0, 0, +1.0) * core) & b), 0.0, cmp="gt")
check("pocket rail holds the top: lid pushed -1 back hits it", vol((Pos(0, 0, -1.0) * core) & b), 0.0, cmp="gt")
check("top edge sits 0.2 under the roof", YI1 - core.bounding_box().max.Y, LIP_GAP, tol=0.01)
check("bottom edge flush with the floor face", core.bounding_box().min.Y, Y0, tol=0.01)
check("lens tip stands LENS_PROUD of the lid face", parts["openmv_n6"].bounding_box().max.Z - Z_TOP, LENS_PROUD, tol=0.01)
check("lens lock ring stays behind the lid (z gap)", Z_CEIL - (N6.ZT + N6.LOCKRING_Z[1]), 0.5, cmp="ge")
for x in LID_SCREW_X:
    rod = Pos(x, Y0 - 3.0, LID_SCREW_Z) * Rot(-90, 0, 0) * Cylinder(2.6 / 2, 3.0 + WALL + 1.0,
                                                                   align=(Align.CENTER, Align.CENTER, Align.MIN))
    check(f"optional floor screw at x={x:.0f} passes the floor wall", vol(rod & b), 0.0)
    bite = Pos(x, YI0 + LIP_GAP, LID_SCREW_Z) * Rot(-90, 0, 0) * Cylinder(2.6 / 2, LIP_T + LID_PILOT_DEPTH,
                                                                           align=(Align.CENTER, Align.CENTER, Align.MIN))
    check(f"optional floor screw at x={x:.0f} bites the lid", vol(bite & core), 0.0, cmp="gt")
    for name, p in parts.items():
        check(f"lid block at x={x:.0f} clears {name}",
              sum(vol(box((x - LID_BLOCK[0] / 2, YI0, Z_CEIL - LID_BLOCK[2]), (LID_BLOCK[0], LID_BLOCK[1] + LIP_T + 1, LID_BLOCK[2])) & sld)
                  for sld in solids_of(p)), 0.0)
# the hood must grow out of the full roof wall, not sit on the roof chamfer's neck
root = box((XI0 + 5.0, YI1 - 0.01, Z_CEIL - CHAMFER_ROOF), (XI1 - XI0 - 10.0, WALL + 0.02, CHAMFER_ROOF))
check("hood root is solid across the full roof wall thickness", vol(root & b) / vol(root), 1.0, tol=0.02)
check("no exposed face behind the hood at the lid plane",
      len([f for f in b.faces() if abs(f.center().Z - (Z_CEIL - 0.01)) < 0.05 and f.center().Y > YI1]), 0)
check("no lid screw or counterbore on the front face (flat outer face)",
      len([f for f in l.faces() if abs(f.center().Z - Z_TOP) < 1e-3 and f.area > 1.0]), 1)
check("DFR0535 parts stay behind the N6 tails (z gap)",
      -3.0 - parts["dfr0535"].bounding_box().max.Z, 1.0, cmp="ge")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
    sys.exit(1)
print("all checks passed")
