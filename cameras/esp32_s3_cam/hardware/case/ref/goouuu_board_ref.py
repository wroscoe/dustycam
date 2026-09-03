"""GOOUUU ESP32-S3-CAM (sargineer product goouuu-esp32s3cam) - envelope v2, photo-measured.

Frame: origin PCB plan bottom-left, +X along the long edge (USB-C ports on the X=0 end,
WROOM antenna overhanging the +X end), Z=0 PCB bottom, camera lens looks +Z.

Dimensions were scaled off the Keyestudio MB0184 ESP32-S3 CAM drawing/photos (same
CH340 + OV3660 + 2x USB-C 40-pin board) using the 2.54 mm header pitch as the ruler.
Treat every figure as +/-0.5 mm (heights +/-1) until the owned unit is calipered.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent                # .../hardware/case/ref
sys.path[:0] = [str(HERE), str(HERE.parents[3] / "hardware_common")]   # cameras/hardware_common/pcbkit.py
from pcbkit import *

PCB_L, PCB_W, PCB_T = 63.0, 29.0, 1.6        # top-view photo 62.6-62.9 (bottom-view read 60.0); DevKitC-1 class
PCB_R = 1.5
HOLES = [(2.4, 2.0), (2.4, 27.0)]            # two plated holes at the USB-end corners, dia ~2.4
HOLE_D = 2.4
PIN_X0, PIN_N, PITCH = 11.0, 20, 2.54        # first pin centre x, pins per row
ROW_Y = (1.8, 27.2)                          # 25.4 mm row spacing (breadboard)
HDR_PLASTIC, PIN_BELOW, PIN_ABOVE = 2.5, 6.0, 1.0   # plastic on the underside, pins point down
USB = [dict(y0=4.2, w=8.8), dict(y0=15.6, w=8.8)]    # TTL (CH340) then OTG; shell x -1.3..5.9, 3.3 tall
USB_X0, USB_L, USB_H = -1.3, 7.2, 3.3
BUTTONS = [(12.0, 4.6, 4.5, 3.5, 3.5), (12.0, 20.9, 4.5, 3.5, 3.5)]   # BOOT, RST: x0,y0,w,d,h
LED = (24.0, 5.7)                            # WS2812 on GPIO48
FPC = (31.0, 6.0, 9.0, 16.5, 2.8)            # 24p 0.5 mm camera connector x0,y0,w,d,h
MODULE = dict(x0=42.5, y0=5.5, w=18.0, l=25.5, can_l=19.2, h=3.1, ant_h=0.9)   # ESP32-S3-WROOM-1, antenna overhangs +X
CAM_C = (46.2, 15.0)                         # OV3660 head parked on the shield can, lens centre
CAM_HEAD, CAM_H = 9.0, 5.4                   # head footprint, height above the can top
LENS_D = 7.0
TOP_H = MODULE["h"] + CAM_H                  # 8.5 tallest thing above the PCB top (the lens)
TF = (44.0, 6.5, 16.0, 16.0, 1.9)            # microSD push slot on the underside; card enters from +X (unverified)
ANT_END = MODULE["x0"] + MODULE["l"]         # 68.0
PIN_XS = [PIN_X0 + PITCH * k for k in range(PIN_N)]
PIN_PTS = [(x, y) for y in ROW_Y for x in PIN_XS]


def gen_step():
    pcb = slab(PCB_L, PCB_W, PCB_T, r=PCB_R, label="pcb", color=BLACK)
    pcb = drill(pcb, [(x, y, HOLE_D) for (x, y) in HOLES] + [(x, y, 1.0) for (x, y) in PIN_PTS])
    parts = [pcb]
    hdr = None
    for y in ROW_Y:
        s = box((PIN_X0 - PITCH / 2, y - PITCH / 2, -HDR_PLASTIC), (PITCH * PIN_N, PITCH, HDR_PLASTIC))
        hdr = s if hdr is None else hdr + s
    for (x, y) in PIN_PTS:
        hdr = hdr + box((x - 0.32, y - 0.32, -HDR_PLASTIC - PIN_BELOW), (0.64, 0.64, HDR_PLASTIC + PIN_BELOW + PCB_T + PIN_ABOVE))
    hdr.label, hdr.color = "pin_headers_2x20", BLACK
    parts.append(hdr)
    for i, u in enumerate(USB):
        parts.append(box((USB_X0, u["y0"], PCB_T), (USB_L, u["w"], USB_H), label=["usb_c_ttl", "usb_c_otg"][i], color=METAL))
    for i, (x, y, w, d, h) in enumerate(BUTTONS):
        parts.append(box((x, y, PCB_T), (w, d, h), label=["button_boot", "button_rst"][i], color=WHITE))
    parts.append(box((LED[0] - 0.75, LED[1] - 0.75, PCB_T), (1.5, 1.5, 1.0), label="rgb_led", color=WHITE))
    parts.append(box((FPC[0], FPC[1], PCB_T), (FPC[2], FPC[3], FPC[4]), label="camera_fpc_connector", color=WHITE))
    m = MODULE
    can = box((m["x0"], m["y0"], PCB_T), (m["can_l"], m["w"], m["h"]), label="esp32_s3_wroom_1_can", color=METAL)
    ant = box((m["x0"] + m["can_l"], m["y0"], PCB_T), (m["l"] - m["can_l"], m["w"], m["ant_h"]), label="wroom_antenna", color=PCB_COLOR)
    parts += [can, ant]
    zc = PCB_T + m["h"]
    head = cbox(CAM_C, zc, (CAM_HEAD, CAM_HEAD, CAM_H - 1.5), label="ov3660_head", color=BLACK)
    lens = cyl(CAM_C, zc + CAM_H - 1.5, LENS_D, 1.5, label="ov3660_lens", color=BLACK)
    parts += [head, lens]
    parts.append(box((TF[0], TF[1], -TF[4]), (TF[2], TF[3], TF[4]), label="microsd_slot", color=METAL))
    return assembly("goouuu_esp32s3cam_board_envelope_v2", parts)


if __name__ == "__main__":
    from build123d import export_step
    export_step(gen_step(), str(HERE / "goouuu_board_ref.step"))
    print("exported goouuu_board_ref.step")
