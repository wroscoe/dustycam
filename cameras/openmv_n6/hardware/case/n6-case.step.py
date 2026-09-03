"""OpenMV Cam N6 + LiPo case — 3-part printed assembly, shown assembled.

Parts print separately (front-cup.step.py / cam-plate.step.py / back-cup.step.py); this is
the assembled view in the shared N6 board frame.  See README.md.
"""

from build123d import Compound

from caselib import back_cup, cam_plate, front_cup


def gen_step():
    parts = [front_cup(), cam_plate(), back_cup()]
    assembly = Compound(children=parts)
    assembly.label = "openmv_n6_case"
    return assembly


if __name__ == "__main__":
    print(gen_step().bounding_box())
