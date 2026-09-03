"""Review assembly: cup + plate + board mock, all in the board frame."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build123d import Compound
import caselib


def gen_step():
    parts = [caselib.cup(), caselib.plate(), caselib.board_mock()]
    asm = Compound(children=parts)
    asm.label = "rt1062_case_assembly"
    return asm
