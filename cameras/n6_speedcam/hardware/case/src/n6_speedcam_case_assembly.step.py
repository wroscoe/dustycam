"""n6_speedcam case: body + lid + every reference part, in the N6 board frame (+Z = forward)."""
from n6_speedcam_case_common import body, lid, refs
from build123d import Compound


def gen_step():
    asm = Compound(children=[body(), lid(), *refs()])
    asm.label = "n6_speedcam_case_assembly"
    return asm


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
