"""Front cup — printable part.  Orientation: face down (+Z max on the bed)."""

from caselib import front_cup


def gen_step():
    return front_cup()


if __name__ == "__main__":
    print(gen_step().bounding_box())
