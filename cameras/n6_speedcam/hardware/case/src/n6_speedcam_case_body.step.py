"""n6_speedcam case body (back). Parameters + geometry: n6_speedcam_case_common.py."""
from n6_speedcam_case_common import body


def gen_step():
    return body()


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
