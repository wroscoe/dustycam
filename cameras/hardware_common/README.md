# hardware_common

Design-source modules shared by more than one camera's `hardware/` tree.

| File | Contents |
|---|---|
| `pcbkit.py` | build123d primitives — `slab`, `box`, `cbox`, `cyl`, `drill`, `pin_headers`, `assembly` — plus the shared colour set. Origin conventions: mm, Z up, board origin at the plan bottom-left, Z=0 at the PCB bottom. |
| `caseskit.py` | Enclosure features shared by the camera cases: the deployed-pose axis handling, `tripod_boss_z`, `drip_ring`, `port_slot`, `membrane_hole`, `window_with_rebate`, `brow`. The weather strategy is documented in its module docstring. |

The repo's ownership rule is that each device owns its own `hardware/`. This
directory is the deliberate exception: it exists only for source that two or
more devices genuinely share, so a fix lands once. Anything used by exactly one
camera belongs in that camera's tree, not here.

`pcbkit.py` came from the sargineer part bundles and is kept byte-compatible
with them so board reference models drop in unmodified.
