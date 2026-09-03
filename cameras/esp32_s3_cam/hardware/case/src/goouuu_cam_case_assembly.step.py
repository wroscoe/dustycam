"""GOOUUU case + board, assembled in the board frame."""
from goouuu_cam_case_common import body, lid, board_ref
from build123d import Compound


def gen_step():
    return Compound(label="goouuu_cam_case_assembly",
                    children=[body(), lid(), board_ref()])


if __name__ == "__main__":
    print("bbox:", gen_step().bounding_box())
