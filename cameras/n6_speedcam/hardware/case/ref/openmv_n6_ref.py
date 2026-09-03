"""OpenMV Cam N6 reference model for the n6_speedcam case.

Verbatim copy of ~/cad/openmv-n6/openmv-n6.py (2026-08-12), measured from
OpenMV's own GLB models -- see the provenance block below and
../README.md. Kept here so the case tree is self-contained; if the two
ever diverge, this copy is the one the interference checks run against.
"""

from build123d import *

# --------------------------------------------------------------------------
# PCB
# --------------------------------------------------------------------------
PCB_W = 35.56          # 1.400 in
PCB_L = 44.45          # 1.750 in
PCB_T = 1.30
ZT = PCB_T             # PCB top face — datum for every component below

# Mounting holes (gold-ringed, user mounting per datasheet "2 mounting holes")
MOUNT_HOLE_D = 2.80
MOUNT_HOLES = [(3.048, 41.402), (32.512, 41.402)]   # 29.464 mm apart

# Camera-module standoff holes (Wurth 9774030243R SMT spacers press in here)
STANDOFF_HOLE_D = 3.00
STANDOFFS = [(2.540, 36.195), (33.020, 36.195)]     # 30.480 mm apart

# 2 x 2x8 female headers, 2.54 mm pitch
HDR_HOLE_D = 1.02
HDR_COLS_X = [1.600, 4.140, 31.460, 34.000]         # L1, L2, R1, R2
HDR_Y0 = 1.599
HDR_PITCH = 2.54
HDR_ROWS = 8
HDR_BODY_H = 8.50
HDR_TAIL = 3.00                                     # assumed
HDR_BLOCKS = [(0.33, 5.41), (30.19, 35.27)]         # x span of each 2-row block
HDR_Y_SPAN = (0.32, 20.64)

# Board cutouts around the mid-mount connectors (x0, y0, dx, dy)
CUTOUTS = [
    (8.600, 6.000, 2.800, 0.800),    # LiPo connector body relief
    (19.053, 2.170, 0.600, 1.400),   # USB-C hold-down leg
    (27.693, 2.170, 0.600, 1.400),   # USB-C hold-down leg
    (19.053, 6.200, 2.054, 1.700),   # USB-C shell tab
    (26.239, 6.200, 2.054, 1.700),   # USB-C shell tab
]

# --------------------------------------------------------------------------
# Optics — M12 x 0.5 (S-mount), 2.8 mm f/2.0 lens as shipped
# --------------------------------------------------------------------------
LENS_AXIS = (17.81, 36.255)        # optical axis, board coords
HOLDER = (5.91, 27.75, 23.80, 17.00)   # x0, y0, dx, dy
HOLDER_Z = (4.65, 18.95)               # relative to ZT
HOLDER_BORE_D = 14.20
BARREL_D = 14.00
BARREL_Z = (13.25, 29.95)              # top = 29.95 above PCB top (~30 mm spec)
LOCKRING_D = 16.20
LOCKRING_Z = (18.95, 21.55)
FRONT_ELEMENT_D = 9.70
FRONT_ELEMENT_DEPTH = 0.20

# Camera daughter board
CAM_PCB_Z = (3.048, 4.648)             # 1.60 mm thick, on the standoffs
CAM_PCB_BODY = (10.918, 28.635, 15.142, 15.875)
CAM_PCB_ARM = (-0.102, 33.334, 35.814, 5.842)
SCREW_HEAD_D = 4.00
SCREW_HEAD_H = 1.61

# --------------------------------------------------------------------------
# I/O and other components — (x0, y0, dx, dy, z0, dz) with z relative to ZT
# --------------------------------------------------------------------------
USB_C = (18.88, 0.27, 9.58, 7.53, -0.85, 4.16)      # mid-mount, shell only
MICROSD = (23.62, 21.15, 11.40, 11.95, -2.76, 1.45)  # bottom side, +X opening
LIPO = (7.00, 0.20, 6.00, 7.70, 0.00, 4.96)          # 2-pin 3.7 V LiPo
JTAG = (23.89, 14.63, 5.08, 6.35, 0.42, 5.71)        # 10-pin 1.27 mm ARM SWD
BOOT1_SW = (32.75, 21.50, 2.50, 8.00, 0.00, 2.49)
B2B = (11.73, 38.34, 12.10, 4.60, 0.00, 3.048)       # DF12 camera-module socket
BUTTONS = [                                          # side-actuated, overhang +Y
    (6.18, 41.53, 4.60, 3.55, 0.00, 1.43),           # USER (SW)
    (24.80, 41.53, 4.60, 3.55, 0.00, 1.43),          # PWR  (SW2)
]

STANDOFF_OD = 4.35
STANDOFF_Z = (-1.39, 3.048)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
MIN3 = (Align.MIN, Align.MIN, Align.MIN)
CCMIN = (Align.CENTER, Align.CENTER, Align.MIN)


def box_at(x0, y0, z0, dx, dy, dz):
    return Pos(x0, y0, z0) * Box(dx, dy, dz, align=MIN3)


def comp_box(spec, label):
    """Component box from an (x0, y0, dx, dy, z0, dz) spec, z relative to ZT."""
    x0, y0, dx, dy, z0, dz = spec
    part = box_at(x0, y0, ZT + z0, dx, dy, dz)
    part.label = label
    return part


def cyl_at(cx, cy, z0, d, h):
    return Pos(cx, cy, z0) * Cylinder(d / 2, h, align=CCMIN)


def rounded_prism(x0, y0, dx, dy, z0, dz, radius):
    sketch = Pos(x0 + dx / 2, y0 + dy / 2) * RectangleRounded(dx, dy, radius)
    return Pos(0, 0, z0) * extrude(sketch, amount=dz)


def header_pin_xy():
    for x in HDR_COLS_X:
        for row in range(HDR_ROWS):
            yield x, HDR_Y0 + row * HDR_PITCH


# --------------------------------------------------------------------------
def gen_step():
    parts = []

    # ---- main PCB ----------------------------------------------------------
    pcb = box_at(0, 0, 0, PCB_W, PCB_L, PCB_T)
    thru = PCB_T + 2.0
    for cx, cy in MOUNT_HOLES:
        pcb -= cyl_at(cx, cy, -1.0, MOUNT_HOLE_D, thru)
    for cx, cy in STANDOFFS:
        pcb -= cyl_at(cx, cy, -1.0, STANDOFF_HOLE_D, thru)
    for cx, cy in header_pin_xy():
        pcb -= cyl_at(cx, cy, -1.0, HDR_HOLE_D, thru)
    for x0, y0, dx, dy in CUTOUTS:
        pcb -= box_at(x0, y0, -1.0, dx, dy, thru)
    pcb.label = "pcb_main"
    parts.append(pcb)

    # ---- headers: 2 x 2x8 female, 8.50 mm above the PCB --------------------
    for i, (x0, x1) in enumerate(HDR_BLOCKS):
        body = box_at(x0, HDR_Y_SPAN[0], ZT,
                      x1 - x0, HDR_Y_SPAN[1] - HDR_Y_SPAN[0], HDR_BODY_H)
        body.label = f"header_{'left' if i == 0 else 'right'}_2x8"
        parts.append(body)

    tails = None
    for cx, cy in header_pin_xy():
        pin = Pos(cx, cy, -HDR_TAIL) * Box(0.64, 0.64, HDR_TAIL + ZT,
                                           align=CCMIN)
        tails = pin if tails is None else tails + pin
    tails.label = "header_tails"
    parts.append(tails)

    # ---- I/O ---------------------------------------------------------------
    usb = comp_box(USB_C, "usb_c_receptacle")
    usb = fillet(usb.edges().filter_by(Axis.Y), 1.50)
    usb.label = "usb_c_receptacle"
    parts.append(usb)

    parts.append(comp_box(MICROSD, "microsd_socket_bottom"))
    parts.append(comp_box(LIPO, "lipo_battery_connector"))
    parts.append(comp_box(JTAG, "jtag_swd_header_2x5"))
    parts.append(comp_box(BOOT1_SW, "boot1_switch"))
    parts.append(comp_box(B2B, "camera_b2b_connector"))
    for name, spec in zip(("button_user_sw", "button_pwr_sw2"), BUTTONS):
        parts.append(comp_box(spec, name))

    # ---- camera-module standoffs and screws --------------------------------
    posts = None
    heads = None
    for cx, cy in STANDOFFS:
        post = cyl_at(cx, cy, ZT + STANDOFF_Z[0], STANDOFF_OD,
                      STANDOFF_Z[1] - STANDOFF_Z[0])
        posts = post if posts is None else posts + post
        head = cyl_at(cx, cy, ZT + CAM_PCB_Z[1], SCREW_HEAD_D, SCREW_HEAD_H)
        heads = head if heads is None else heads + head
    posts.label = "camera_standoffs"
    heads.label = "camera_screw_heads"
    parts.extend([posts, heads])

    # ---- camera daughter board (cross-shaped) ------------------------------
    cam_t = CAM_PCB_Z[1] - CAM_PCB_Z[0]
    cam = (box_at(*CAM_PCB_BODY[:2], ZT + CAM_PCB_Z[0], *CAM_PCB_BODY[2:], cam_t)
           + box_at(*CAM_PCB_ARM[:2], ZT + CAM_PCB_Z[0], *CAM_PCB_ARM[2:], cam_t))
    for cx, cy in STANDOFFS:
        cam -= cyl_at(cx, cy, ZT + CAM_PCB_Z[0] - 1.0, 2.20, cam_t + 2.0)
    cam.label = "pcb_camera_module"
    parts.append(cam)

    # ---- M12 lens holder ---------------------------------------------------
    holder = rounded_prism(HOLDER[0], HOLDER[1], HOLDER[2], HOLDER[3],
                           ZT + HOLDER_Z[0], HOLDER_Z[1] - HOLDER_Z[0], 2.00)
    holder -= cyl_at(*LENS_AXIS, ZT + HOLDER_Z[0] - 1.0, HOLDER_BORE_D,
                     HOLDER_Z[1] - HOLDER_Z[0] + 2.0)
    holder.label = "lens_holder_m12"
    parts.append(holder)

    # ---- M12 lens ----------------------------------------------------------
    barrel = cyl_at(*LENS_AXIS, ZT + BARREL_Z[0], BARREL_D,
                    BARREL_Z[1] - BARREL_Z[0])
    barrel += cyl_at(*LENS_AXIS, ZT + LOCKRING_Z[0], LOCKRING_D,
                     LOCKRING_Z[1] - LOCKRING_Z[0])
    barrel -= cyl_at(*LENS_AXIS, ZT + BARREL_Z[1] - FRONT_ELEMENT_DEPTH,
                     FRONT_ELEMENT_D, FRONT_ELEMENT_DEPTH + 1.0)
    barrel.label = "lens_m12_2p8mm"
    parts.append(barrel)

    assembly = Compound(children=parts)
    assembly.label = "openmv_n6"
    return assembly


if __name__ == "__main__":
    shape = gen_step()
    print(shape.bounding_box())
