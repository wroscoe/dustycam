# openmv_n6 hardware

No board-specific design work yet — the OpenMV N6 is used on its stock board.

## Enclosure: blocked on mechanical data

**Update 2026-09-02: the mechanical data exists.** The N6 was measured from
OpenMV's own GLB models on 2026-08-12 (`~/cad/openmv-n6/DIMENSIONS.md`), and
a copy of both the dimension notes and the build123d reference model now
lives in the repo at
[`../../n6_speedcam/hardware/case/ref/`](../../n6_speedcam/hardware/case/ref/)
(`OPENMV_N6_DIMENSIONS.md`, `openmv_n6_ref.py`). The n6_speedcam case hangs
the N6 from its two Ø2.80 mounting holes; a plain N6-only case for this
directory can reuse that reference model and the same boss recipe. The rest
of this note is the pre-measurement state, kept for the list of what an
enclosure needs.

An enclosure matching the ESP32-S3 cases
([`../../esp32_s3_cam/hardware/case/`](../../esp32_s3_cam/hardware/case/)) is
wanted, but at the time of writing **there was no mechanical data for this board
in this repo or in sarg**. A full 128-part catalogue sweep found no OpenMV
board of any generation (N6, H7, RT1062) from any owner; OpenMV is not a
vendor in the catalogue at all. The 76 OpenMV notes are entirely firmware and
sensor behaviour, with nothing mechanical.

So this one starts from a caliper, not from a model. To unblock it, capture —
either from the vendor's published hardware files or by measurement:

- PCB outline L x W x thickness, corner radius
- mounting hole centres + diameter
- USB-C shell size, its centre offset from the board datum, protrusion past
  the PCB edge, and **which edge it is on** (that edge becomes the floor)
- tallest component above and below the PCB
- M12 lens barrel diameter, its axis position, and height above the PCB
- any connector that must stay reachable, and the SD slot position

Then the case is the same recipe as the ESP32-S3 pair: stand it on the
connector edge, ports down, tripod insert on the back, and reuse
`cameras/hardware_common/caseskit.py`.

Worth noting for whoever does it: the low-power variant needs *"first boot
near USB"* supervision, so convenient USB access matters more on this board,
not less.

See [`../../pi5cam/hardware/`](../../pi5cam/hardware/) for the conventions
(design source in git, exports regenerated under `export/`).
