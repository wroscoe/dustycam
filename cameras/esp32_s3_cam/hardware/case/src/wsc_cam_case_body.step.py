"""Waveshare ESP32-S3-CAM-GC0308 case body. Parameters: wsc_cam_case_common.py."""
from wsc_cam_case_common import body


def gen_step():
    return body()


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
