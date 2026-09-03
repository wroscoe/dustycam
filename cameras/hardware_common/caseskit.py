"""Shared enclosure features for the dustycam camera cases.

Units mm, Z up, authored in the *board* frame (see pcbkit.py conventions).

These helpers exist because all the small-camera cases share one deployment
pose and one weather strategy, and both should be written once.

**Pose.** Each case is designed flat in the board frame, then *stood on the
connector edge* in the field: the wall carrying the USB/battery connectors
becomes the floor, the lid becomes the forward-facing front, and the camera
looks out horizontally. That puts the ports on the bottom -- which is what was
asked for, and independently the right call outdoors, because a
downward-facing opening cannot pool water or catch falling rain.

Because the connector wall differs per board, the deployed "down" direction
differs too, and the helpers take it as an axis name:

    GOOUUU  DOWN = "-X"   (USB-C ports sit on the board's X=0 end)
    Waveshare DOWN = "-Y" (underside USB-C sits on the board's Y=0 edge)

The lid is the +Z face on both boards, so window/membrane/brow features are
always authored on Z and only need to know which way is up.

**Splash resistance (no gasket).** The rules implemented here:
  * every opening faces down, or is shielded by a brow;
  * seams shed outward -- a proud drip rib breaks the path water would
    otherwise wick around an edge and into a joint;
  * holes that need not pass a plug are blind, closed by a thin printed
    membrane (buttons stay pressable, an LED still shows through);
  * the interior has a drain at its low corner so condensate leaves.

This is not a sealed enclosure. It is "survives rain under an eave", matching
the pi5cam case README's PETG/ASA guidance.
"""
from build123d import *
from pcbkit import box, cyl

# axis name -> (index, sign of the outward normal)
_AXIS = {"-X": (0, -1), "+X": (0, +1), "-Y": (1, -1), "+Y": (1, +1),
         "-Z": (2, -1), "+Z": (2, +1)}


def _span(axis):
    """The two in-plane axis indices for a face whose normal is `axis`."""
    i = _AXIS[axis][0]
    return [k for k in (0, 1, 2) if k != i]


def fillet_safe(part, r, axis=Axis.Z):
    """Fillet edges parallel to `axis` when the geometry allows; skip if not."""
    try:
        return fillet(part.edges().filter_by(axis), r)
    except Exception:
        return part


def tripod_boss_z(part, center_xy, z_face, z_inner, proud, inner_h,
                  boss_d=16.0, hole_d=8.0, hole_depth=13.5, back_wall=1.0):
    """1/4"-20 heat-set insert boss on the -Z face (the deployed camera's BACK).

    The insert (ruthex RX-1/4-20, 12.7 long) presses in from the *outside*, so
    the pocket is blind and never opens into the cavity. Axial material splits
    between an external pad and an internal column, keeping the external bump
    small wherever the interior has room for the column:

        proud (below z_face) + floor (z_inner - z_face) + inner_h

    Pass inner_h=0 where the interior is too crowded to take a column; the pad
    then carries the whole insert.
    """
    need = hole_depth + back_wall
    have = proud + (z_inner - z_face) + inner_h
    if have < need - 1e-6:
        raise ValueError(
            f"tripod boss too shallow: {have:.2f} mm of material for a "
            f"{hole_depth} mm insert + {back_wall} mm back wall "
            f"(need {need:.2f}). Increase proud or inner_h.")
    part = part + cyl(center_xy, z_face - proud, boss_d, proud + 0.01)
    if inner_h > 0:
        part = part + cyl(center_xy, z_inner, boss_d, inner_h)
    part = part - cyl(center_xy, z_face - proud - 0.01, hole_d, hole_depth + 0.01)
    return part


def face_chamfer(part, face, size):
    """Chamfer the perimeter of one face -- the angled replacement for a rib.

    A drop crossing a sharp arris wraps around it and tracks back along the
    underside; a chamfered arris breaks it off instead. That is the whole job
    the old proud drip ring did on the port face, done with one angle rather
    than a four-box loop stuck to the outside, and it costs no extra height.

    Applied to the roof face it does the same for the top edges, replacing the
    milled rain gutter -- which only ever helped water running one direction
    and left 1.4 mm of wall where it cut.

    Call this BEFORE cutting the port openings, so only the perimeter edges are
    in the group. Raises if the chamfer is not constructible; that is a real
    geometry problem, not something to swallow.
    """
    i, sign = _AXIS[face]
    axis = [Axis.X, Axis.Y, Axis.Z][i]
    groups = part.edges().group_by(axis)
    return chamfer(groups[-1] if sign > 0 else groups[0], size)


def port_slot(part, lo, hi, face, center_a, center_b, w_a, w_b,
              wall_t, lead_in=0.6):
    """A rectangular opening through the deployed-down wall, with a lead-in.

    `center_a`/`w_a` run along the first in-plane axis of the face, `center_b`
    /`w_b` along the second (see `_span`). The mouth is flared by `lead_in` on
    the outside so a plug finds it and so the edge sheds rather than holds a
    drop.
    """
    i, sign = _AXIS[face]
    a, b = _span(face)
    base = hi[i] if sign > 0 else lo[i]

    def cut(pad, depth, start):
        at, size = [0, 0, 0], [0, 0, 0]
        at[a], size[a] = center_a - w_a / 2 - pad, w_a + 2 * pad
        at[b], size[b] = center_b - w_b / 2 - pad, w_b + 2 * pad
        at[i], size[i] = start, depth
        return box(tuple(at), tuple(size))

    # Through-cut: generous overshoot on the outside so it also clears anything
    # already standing proud of the face (the drip ring, a tripod pad).
    if sign < 0:
        part = part - cut(0.0, wall_t + 6.0, base - 5.0)
        part = part - cut(lead_in, 5.0 + lead_in, base - 5.0)
    else:
        part = part - cut(0.0, wall_t + 6.0, base - wall_t - 1.0)
        part = part - cut(lead_in, 5.0 + lead_in, base - lead_in)
    return part


def membrane_hole(part, center_xy, d, z_outer, z_inner, membrane=0.6):
    """A hole stopped short by a thin printed membrane, cut from the inside.

    A button behind ~0.6 mm of PETG still presses; an LED behind 0.8 mm still
    shows. Neither admits water, and neither needs a separate seal. The
    membrane lands flush with the outer surface.
    """
    lo, hi = min(z_outer, z_inner), max(z_outer, z_inner)
    if z_outer > z_inner:                       # outer face is the +Z side
        return part - cyl(center_xy, lo - 0.01, d, (hi - lo) - membrane + 0.01)
    return part - cyl(center_xy, lo + membrane, d, (hi - lo) - membrane + 0.01)


def conical_window(part, center_xy, z_outer, z_inner, d_inner, d_outer):
    """A camera window that opens out conically toward the outer face.

    One angled cut doing three jobs the old cylinder-plus-rebate needed three
    for: the flare is field-of-view relief, so the window can be tight on the
    lens without vignetting; the sloped wall sheds water instead of holding a
    ring of it against the glass; and there is no square shelf left over for a
    disc that has not been sourced.
    """
    cx, cy = center_xy
    t = abs(z_outer - z_inner)
    lo = min(z_outer, z_inner)
    r_in, r_out = d_inner / 2, d_outer / 2
    # overshoot both faces along the same taper so nothing is left uncut
    slope = (r_out - r_in) / t
    cone = Pos(cx, cy, lo - 1.0) * Cone(
        bottom_radius=r_in - slope, top_radius=r_out + slope, height=t + 2.0,
        align=(Align.CENTER, Align.CENTER, Align.MIN))
    if z_outer < z_inner:                      # outer face is the -Z side
        cone = Pos(cx, cy, 2 * lo + t) * Rot(180, 0, 0) * Pos(-cx, -cy, 0) * cone
    return part - cone


def visor(part, z_face, up, center_across, width, above, proj=6.0, height=6.0):
    """A single wedge shading the window -- the angled replacement for a brow.

    The old brow was a flat shelf plus a turned-down lip: two boxes, a square
    underside for water to cling to and run back along, and a lip that had to
    be kept clear of the field of view. This is one triangular prism whose
    sloped top throws rain forward off the tip, with nothing hanging below the
    line it starts from -- so `above` only has to clear the window itself.

    `up` is the deployed-up axis ("+X" or "+Y"); `above` is where the wedge
    starts on it; the wedge rises `height` up the lid face and projects `proj`.
    """
    i, sign = _AXIS[up]
    if sign < 0:
        raise ValueError("visor expects a positive up axis ('+X' or '+Y')")
    # triangle in the (up, Z) plane: at the face, from `above` up to
    # `above + height`, tapering to a tip `proj` out at `above`
    pts = [(above, z_face), (above + height, z_face), (above, z_face + proj)]
    plane = Plane.XZ if i == 0 else Plane.YZ
    wedge = extrude(plane * Polygon(*pts, align=None), amount=width / 2, both=True)
    offset = [0.0, 0.0, 0.0]
    offset[1 if i == 0 else 0] = center_across
    return part + Pos(*offset) * wedge


def brow(part, z_face, up, center_across, width, above,
         proj=5.0, thick=2.0, drop=2.0):
    """A rain brow standing off the lid, above a feature in the deployed sense.

    `up` is the deployed-up axis name ("+X" or "+Y"); `above` is that axis'
    coordinate for the shelf's underside; `center_across`/`width` place it on
    the other in-plane axis. The turned-down lip at its outer edge throws
    water clear instead of letting it run back along the underside.

    The lip hangs `drop` BELOW `above`, so it -- not the shelf -- is what can
    clip a camera's field of view. Callers must pass
    `above >= (top of the window) + drop + margin`.
    """
    i, sign = _AXIS[up]
    if sign < 0:
        raise ValueError("brow expects a positive up axis ('+X' or '+Y')")
    across = [k for k in (0, 1) if k != i][0]
    at, size = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    at[across], size[across] = center_across - width / 2, width
    # the shelf: a plate standing out of the lid face, just above the feature
    at[i], size[i] = above, thick
    at[2], size[2] = z_face, proj
    shelf = box(tuple(at), tuple(size))
    # the lip: hangs down off the shelf's outer edge so water drips clear
    # instead of running back along the underside to the window
    at[i], size[i] = above - drop, drop
    at[2], size[2] = z_face + proj - thick, thick
    lip = box(tuple(at), tuple(size))
    return part + shelf + lip


"""Drains are cut with `port_slot` on the same downward wall as the ports --
a drain is just a small opening in the floor, and reusing one axis-aware
routine keeps a second orientation bug from creeping in."""
