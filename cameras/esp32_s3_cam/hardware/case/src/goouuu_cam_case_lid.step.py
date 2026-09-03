"""GOOUUU ESP32-S3-CAM case lid (the deployed FRONT face)."""
from goouuu_cam_case_common import lid


def gen_step():
    return lid()


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
