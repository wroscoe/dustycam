"""n6_speedcam case lid (front: radome, lens window, sensor bosses). See n6_speedcam_case_common.py."""
from n6_speedcam_case_common import lid


def gen_step():
    return lid()


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
