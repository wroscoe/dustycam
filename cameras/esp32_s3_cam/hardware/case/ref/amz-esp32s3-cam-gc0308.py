"""Waveshare ESP32-S3-CAM-GC0308 AIoT camera board (amz B0GS1CMNX5) — envelope.

Frame: origin PCB plan bottom-left, Z=0 PCB bottom. 37 x 37 PCB, R2.25
corners, 4x dia 2.5 holes at 32.6 x 32.6 (vendor drawing). Top block =
TF slot + seated GC0308 camera head to 7.5 (est); bottom blocks =
underside USB-C at bottom edge (3.3 tall, 1.3 overhang, est) and a
connector envelope for the GH1.25/FPC field.
"""
from build123d import *
L, W, T = 37.0, 37.0, 1.6
R = 2.25
HOLE_D = 2.5
HX, HY = 32.6, 32.6
IX, IY = (L-HX)/2, (W-HY)/2
with BuildPart() as p:
    with BuildSketch(Plane.XY):
        with Locations((L/2, W/2)):
            RectangleRounded(L, W, R)
        with Locations(*[(IX+sx*HX, IY+sy*HY) for sx in (0,1) for sy in (0,1)]):
            Circle(HOLE_D/2, mode=Mode.SUBTRACT)
    extrude(amount=T)
    # top: TF slot + camera head parked on it (drawing: slot upper center)
    with BuildSketch(Plane.XY.offset(T)):
        with Locations((L/2 + 1.5, 24.0)):
            Rectangle(15.0, 15.5)
    extrude(amount=7.5)
    # bottom: USB-C shell at bottom edge center, underside mount
    with BuildSketch(Plane.XY):
        with Locations((18.5, 7.35/2 - 1.3)):
            Rectangle(9.0, 7.35)
    extrude(amount=-3.3)
    # bottom: envelope for GH1.25 / FPC / header connector field
    with BuildSketch(Plane.XY):
        with Locations((L/2, W/2 + 4.0)):
            Rectangle(30.0, 24.0)
    extrude(amount=-2.9)
    # Keep-outs at the mounting holes. The 30 x 24 connector field above is a
    # coarse "somewhere in here" envelope read off ref/interface-map.jpg, and as
    # drawn it laps over two of the board's own M2 holes -- which the real board
    # cannot do, since those holes have to take a screw and a standoff. Clearing
    # a dia 6 column at each hole removes 8.4 mm3 of false interference that
    # otherwise trips the case fit check.  ASSUMPTION, not a measurement: confirm
    # against the board when the caliper pass happens.
    with BuildSketch(Plane.XY):
        with Locations(*[(IX+sx*HX, IY+sy*HY) for sx in (0,1) for sy in (0,1)]):
            Circle(6.0/2)
    extrude(amount=-3.0, mode=Mode.SUBTRACT)
p.part.label = "esp32s3-cam-gc0308"
def gen_step():
    return p.part
