"""Cup — front/side/roof shell with lens hood. Board frame (see caselib)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from caselib import cup


def gen_step():
    return cup()
