# Carrier / Power PCB

KiCad project for the DustyCam carrier board (power input, protection, and
any sensor breakouts that graduate from hand wiring).

## Layout

- KiCad project files (`.kicad_pro`, `.kicad_sch`, `.kicad_pcb`) live here at
  the top level — they are text and diff cleanly in git.
- `export/` — fabrication outputs: gerbers, drill files, and a schematic PDF.
  Regenerate on release; don't hand-edit.

## Scope (draft)

Nothing designed yet. Candidate scope, smallest first:

1. **Power entry board**: barrel jack (size TBD — see `docs/build_guide.md`
   open question), reverse-polarity protection, fuse, buck to 5 V/5 A for
   the Pi 5, screw terminals for the solar panel input.
2. Later: UPS/charging integration, RTC battery, PoE.

Keep the power budget in [`../power/`](../power/) as the sizing input for
regulator and trace-width choices.
