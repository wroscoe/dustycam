"""Plate in print orientation: outer face on the bed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from caselib import plate_print


def gen_step():
    return plate_print()
