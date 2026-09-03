"""Lid + everything that mounts on it (radar, N6, carrier), no body — deployed pose, Z-up.

The view to check the sensor stack: look at it from behind (the open side)."""
from n6_speedcam_case_common import lid, radar_ref, n6_ref, carrier_ref
from build123d import Compound, Rot


def gen_step():
    parts = [lid(), radar_ref(), n6_ref(), carrier_ref()]
    asm = Compound(children=[Rot(90, 0, 0) * p for p in parts])
    asm.label = "n6_speedcam_front_module"
    return asm


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
