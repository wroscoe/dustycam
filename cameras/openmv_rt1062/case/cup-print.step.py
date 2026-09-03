"""Cup in print orientation: standing on its open bottom, Z up."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from caselib import cup_print


def gen_step():
    return cup_print()
