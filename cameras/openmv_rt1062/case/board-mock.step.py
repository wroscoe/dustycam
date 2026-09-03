"""Simplified RT1062 R6 board envelope for interference checks."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from caselib import board_mock


def gen_step():
    return board_mock()
