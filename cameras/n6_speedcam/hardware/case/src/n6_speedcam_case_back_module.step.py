"""Body + everything that mounts in it (DFR0535 solar manager, LiPo pouch envelope), no lid.

Deployed pose, Z-up. The body is drawn see-through so the boards behind the back
wall read from any angle; the parts keep their solid colours."""
from n6_speedcam_case_common import body, dfr_ref, battery_ref
from build123d import Compound, Rot, Color


def gen_step():
    b = body()
    b.color = Color(0.25, 0.45, 0.75, 0.30)
    d = dfr_ref()
    for s in d.children:
        s.color = Color(0.15, 0.15, 0.15)
    parts = [b, d, battery_ref()]
    asm = Compound(children=[Rot(90, 0, 0) * p for p in parts])
    asm.label = "n6_speedcam_back_module"
    return asm


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
