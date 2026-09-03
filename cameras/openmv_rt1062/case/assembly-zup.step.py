"""Review assembly rotated to Z-up (use orientation: lens horizontal, cables down)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build123d import Compound, Axis
import caselib


def gen_step():
    parts = [caselib.cup(), caselib.plate(), caselib.board_mock()]
    parts = [p.rotate(Axis.X, 90) for p in parts]
    asm = Compound(children=parts)
    asm.label = "rt1062_case_assembly_zup"
    return asm
