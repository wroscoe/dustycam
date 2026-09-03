"""Cam plate — printable part.  Orientation: flat, bosses up."""

from caselib import cam_plate


def gen_step():
    return cam_plate()


if __name__ == "__main__":
    print(gen_step().bounding_box())
