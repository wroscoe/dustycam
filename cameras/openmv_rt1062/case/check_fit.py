"""Interference + envelope checks. Run with the CAD skill venv python."""
import caselib as C
from build123d import *

cup, plate, mock, plug = C.cup(), C.plate(), C.board_mock(), C.usb_plug_mock()
def vol(s):
    try: return round(s.volume, 3)
    except Exception: return 0.0
print("cup   vol cm3", round(cup.volume/1000, 2), "bb", cup.bounding_box())
print("plate vol cm3", round(plate.volume/1000, 2), "bb", plate.bounding_box())
print("cup solids", len(cup.solids()), "plate solids", len(plate.solids()))
print("cup∩mock  mm3", vol(cup & mock))
print("plate∩mock mm3", vol(plate & mock))
print("cup∩plate mm3", vol(cup & plate))
print("cup∩usbplug mm3", vol(cup & plug))
print("plate∩usbplug mm3", vol(plate & plug))
# swept insertion: board mock moved back along Z by up to 40 mm must not hit the cup
worst = 0
for dz in range(1, 41, 2):
    v = vol(cup & mock.moved(Location((0, 0, -dz))))
    worst = max(worst, v)
print("cup∩mock during slide-in (max mm3)", worst)
# open bottom: nothing of the cup below the board within the cable zone
zone = C.box(C.JST_PLUG_X[0], C.USB_PLUG_X[1], C.Y_BOT - 0.01, 0.0, C.Z_PLATE_IN, C.Z_FRONT_IN)
print("cup in cable zone mm3", vol(cup & zone), "plate in cable zone mm3", vol(plate & zone))
