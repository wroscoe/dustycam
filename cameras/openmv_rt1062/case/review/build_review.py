"""Assemble the review page: template + latest snapshot PNGs as data URIs."""
import base64, glob, json, os, re
VIEWS = [
 ("v-iso",      "Assembly, front-left", "az 35 · el 25",  "Lens hood on the front face, two switch chimneys on the roof, 2 mm drip chamfers, roof lip over the back plate."),
 ("v-front",    "Front",                "front",          "Ø15 lens hole inside the Ø21 hood; the hood ring is full at the wall and cut back 45° underneath toward the tip."),
 ("v-right",    "Right side",           "right",          "Depth 52.3 from hood tip to roof lip; back plate flush with the side walls; open bottom at the base."),
 ("v-back",     "Back",                 "back",           "Back plate: four counterbored M2 screws into the cup's corner blocks, 1/4-20 insert bore (Ø8.8) top-centre, roof drip lip above."),
 ("v-top",      "Top",                  "top",            "Flat bridged roof with chamfers; chimneys sit over the two edge switches; hood projects forward."),
 ("v-bottom",   "From below",           "az 35 · el −40", "Open bottom: USB-C and JST hang straight down between the corner blocks; board edge, bottom blocks and plate rim visible."),
 ("v-iso-xray", "X-ray",                "az 35 · el 25 · transparent", "Board mock inside: lens through the hood, camera module and M12 holder under the roof, plate posts and the inward insert boss behind the board."),
 ("v-cup-back", "Cup from behind (print orientation)", "az 215 · el 35", "Standing as printed. Interior: corner screw blocks, microSD groove in the right wall (left in this view), rim seat, hood beyond the roof."),
 ("v-cup-front","Cup front (print orientation)", "az 35 · el 20", "Bed contact is the bottom edge and the four blocks; walls vertical; hood underside at 45°; Ø15 horizontal lens hole."),
 ("v-plate",    "Back plate (print orientation)", "az 30 · el 40", "Outer face on the bed. Two Ø5 M2 posts under the board's mounting holes, two Ø3.4 corner rests, Ø15 insert boss, U-shaped locating rim, four counterbored holes."),
 ("v-boardmock","Board envelope used for the checks", "az 35 · el 25", "Simplified R6: PCB, camera module, M12 holder + Ø14 barrel, USB-C, JST, microSD (+ card proud), SWD header, edge switches, spacers and screw heads."),
]
def latest(prefix):
    fs = sorted(glob.glob(f"{prefix}_*.png")); return fs[-1]
views = []
for vid, title, cam, note in VIEWS:
    f = latest(vid)
    b = base64.b64encode(open(f, "rb").read()).decode()
    views.append({"id": vid, "title": title, "cam": cam, "note": note, "src": f"data:image/png;base64,{b}"})
tpl = open("template.html").read()
out = tpl.replace("<!--VIEWS-->", "").replace("<script>\n(() => {", "<script>\nwindow.__VIEWS__ = " + json.dumps(views) + ";\n(() => {", 1)
open("rt1062-case-review.html", "w").write(out)
print("wrote rt1062-case-review.html", round(os.path.getsize("rt1062-case-review.html")/1e6, 2), "MB", len(views), "views")
