"""Fit constants and press-fit features for printed enclosures.

Reconstructed 2026-09-02: the original home of these values was
~/tools/mywarehouse/lib/fits.py (documented in tolerances.md), retired and
deleted in the 2026-08 consolidation.  Constants come from this project's
README table; edge_crush_rib() was reverse-engineered from the rib geometry
in the previously generated back-cup.step (slice area 1.2475 mm^2, crush
volume 2.808 mm^3/rib beyond the 0.15 lip gap — both reproduced).  The flank
lead-in curve is not byte-identical to the original: diffing against the old
STEP leaves ~0.17 mm^3 slivers at each rib flank, functionally irrelevant.
"""

from build123d import Vector, Wire, extrude, make_face

LIP_GAP = 0.15          # per-side clearance, press-fit lip into mouth
LIP_RIB_PROUD = 0.25    # crush rib stands proud of the lip wall
PLATE_SLIDE_GAP = 0.20  # per-side clearance, plate sliding into a socket
ELEPHANT_FOOT = 0.40    # chamfer on every bed-contact perimeter, x 45 deg


def edge_crush_rib(height, length=6.0, proud=LIP_RIB_PROUD, lead_ratio=4.0):
    """Vertical crush rib for a press-fit lip.

    Canonical frame: base plane y = 0 lies on the lip outer wall, the rib
    protrudes +Y by `proud`, is centred on x = 0, and runs z = 0..height.
    Cross-section is a trapezoid `length` wide at the wall tapering to
    length - 2*lead_ratio*proud at the crest, so the mating wall shaves the
    crest progressively instead of biting a square shoulder.
    """
    crest = length - 2.0 * lead_ratio * proud
    pts = [
        Vector(-length / 2, 0, 0),
        Vector(length / 2, 0, 0),
        Vector(crest / 2, proud, 0),
        Vector(-crest / 2, proud, 0),
    ]
    face = make_face(Wire.make_polygon(pts, close=True))
    return extrude(face, amount=height)
