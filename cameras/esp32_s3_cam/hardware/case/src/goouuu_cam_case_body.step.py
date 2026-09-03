"""GOOUUU ESP32-S3-CAM case body. Parameters + geometry: goouuu_cam_case_common.py."""
from goouuu_cam_case_common import body


def gen_step():
    return body()


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
