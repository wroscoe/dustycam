"""Shared build123d helpers for the sargparts warehouse generators.

Conventions: millimetres, Z up. Boards/panels: origin at the plan bottom-left
corner, Z=0 at the PCB (or panel back) bottom face. Round parts: axis on Z,
centred at the origin.
"""
from build123d import *

PCB_T = 1.6
PCB_COLOR = Color(0.12, 0.38, 0.22)
METAL = Color(0.72, 0.72, 0.75)
BLACK = Color(0.12, 0.12, 0.12)
WHITE = Color(0.9, 0.9, 0.88)
COPPER = Color(0.80, 0.50, 0.20)


def slab(w, h, t, r=0.0, at=(0.0, 0.0, 0.0), chamfer_c=0.0, label="slab", color=None):
    """Rectangular slab: x in [x0, x0+w], y in [y0, y0+h], z in [z0, z0+t]."""
    sk = RectangleRounded(w, h, r) if r > 0 else Rectangle(w, h)
    part = extrude(sk, t)
    if chamfer_c > 0:
        part = chamfer(part.edges().filter_by(Axis.Z), chamfer_c)
    part = Pos(at[0] + w / 2, at[1] + h / 2, at[2]) * part
    part.label = label
    if color is not None:
        part.color = color
    return part


def drill(part, holes, d_default=2.5):
    """Through-drill (x, y[, d]) holes along Z."""
    label, color = part.label, getattr(part, "color", None)
    for hs in holes:
        d = hs[2] if len(hs) > 2 else d_default
        part = part - Pos(hs[0], hs[1], 0) * Cylinder(d / 2, 200)
    part.label = label
    if color is not None:
        part.color = color
    return part


def box(at, size, label="box", color=None):
    """Box with its min corner at `at`."""
    p = Pos(at[0] + size[0] / 2, at[1] + size[1] / 2, at[2] + size[2] / 2) * Box(*size)
    p.label = label
    if color is not None:
        p.color = color
    return p


def cbox(center_xy, z0, size, label="box", color=None):
    """Box centred in XY on center_xy, sitting on z0."""
    p = Pos(center_xy[0], center_xy[1], z0 + size[2] / 2) * Box(*size)
    p.label = label
    if color is not None:
        p.color = color
    return p


def cyl(center_xy, z0, d, h, label="cyl", color=None):
    p = Pos(center_xy[0], center_xy[1], z0 + h / 2) * Cylinder(d / 2, h)
    p.label = label
    if color is not None:
        p.color = color
    return p


def pin_headers(points, z_top=PCB_T, plastic=2.5, above=6.0, below=3.0, label="pin_headers"):
    """0.1in male pin headers soldered from the top at the given (x, y) points."""
    hdr = None
    for (x, y) in points:
        s = box((x - 1.27, y - 1.27, z_top), (2.54, 2.54, plastic)) + \
            box((x - 0.32, y - 0.32, z_top - below), (0.64, 0.64, below + plastic + above))
        hdr = s if hdr is None else hdr + s
    hdr.label = label
    hdr.color = BLACK
    return hdr


def assembly(label, children):
    return Compound(label=label, children=list(children))
