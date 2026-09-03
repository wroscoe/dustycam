"""Back cup — printable part.  Orientation: back face down (-Z min on the bed)."""

from caselib import back_cup


def gen_step():
    return back_cup()


if __name__ == "__main__":
    print(gen_step().bounding_box())
