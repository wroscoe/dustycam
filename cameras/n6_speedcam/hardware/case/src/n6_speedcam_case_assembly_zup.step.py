"""Same assembly rotated into the DEPLOYED pose for the viewer (Z-up): roof up, lens looking -Y."""
from n6_speedcam_case_common import body, lid, refs
from build123d import Compound, Rot


def gen_step():
    asm = Compound(children=[Rot(90, 0, 0) * p for p in (body(), lid(), *refs())])
    asm.label = "n6_speedcam_case_deployed"
    return asm


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
