"""Back plate — board posts, 1/4-20 insert boss, locating rim, 4 screw holes."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from caselib import plate


def gen_step():
    return plate()
