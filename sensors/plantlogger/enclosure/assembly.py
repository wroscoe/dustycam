"""assembly.py — Stacked (assembled) view of the 3-part plant-logger enclosure.

Imports the three validated part STEPs and places them in the GLOBAL FRAME G
(X=0 inner USB wall face, Y=0 inner long-wall face, Z=0 cavity floor top):

  box       : authored in G  -> identity
  midplate  : local origin = plate corner at G(0.2, 0.2); underside rests on
              gusset tops at Z=6.5           -> translate (+0.2, +0.2, +6.5)
  faceplate : local origin = outer corner at G(-2.4, -2.4); panel underside
              rests on the rim at Z=18.5     -> translate (-2.4, -2.4, +18.5)

Stack check (from the verified spec): gusset top 6.5, midplate 6.5-8.9,
PCB 10.90-11.81, lip bottom 14.5, rim 18.5, panel top 20.9; assembled
outer height 23.3 (Z -2.4 .. 20.9).
"""

from pathlib import Path

from build123d import Compound, Location, import_step

_HERE = Path(__file__).resolve().parent

# Part-local -> GLOBAL FRAME G translations (mm), from the interface spec.
PLACEMENTS = {
    "box": (0.0, 0.0, 0.0),
    "midplate": (0.2, 0.2, 6.5),
    "faceplate": (-2.4, -2.4, 18.5),
}


def gen_step():
    children = []
    for name, offset in PLACEMENTS.items():
        part = import_step(str(_HERE / f"{name}.step"))
        part = part.moved(Location(offset))
        part.label = name
        children.append(part)
    asm = Compound(children=children)
    asm.label = "plant_logger_enclosure_assembly"
    return asm


if __name__ == "__main__":
    asm = gen_step()
    print("bbox:", asm.bounding_box())
