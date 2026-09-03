"""Review-only cutaway: the three case parts (half-sectioned at the lens axis)
with the real N6 board model and the battery mock in place.

NOT a printable artifact — it exists so the fit can be inspected visually and
so board/case interference can be re-checked after any parameter change.
"""

import importlib.util

from build123d import Align, Box, Compound, Pos

import caselib as C

_spec = importlib.util.spec_from_file_location(
    "openmv_n6", str(__import__("pathlib").Path(__file__).resolve().parent / "ref" / "openmv-n6.py")
)
_n6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_n6)


def gen_step():
    cut = Pos(C.CX, C.CY, 4.75) * Box(60, 90, 90, align=(Align.MIN,) * 3)
    cut = Pos(C.CX, -40, -40) * Box(60, 100, 100, align=(Align.MIN,) * 3)

    parts = []
    for fn in (C.front_cup, C.cam_plate, C.back_cup):
        p = fn()
        label = p.label
        p = p - cut
        p.label = label + "_sectioned"
        parts.append(p)

    for child in _n6.gen_step().children:
        parts.append(child)

    parts.append(C.battery_mock())
    parts.append(C.sd_card_mock())
    parts.append(C.usb_plug_mock())
    parts.append(C.battery_cable_mock())

    asm = Compound(children=parts)
    asm.label = "n6_case_fitcheck"
    return asm


if __name__ == "__main__":
    print(gen_step().bounding_box())
