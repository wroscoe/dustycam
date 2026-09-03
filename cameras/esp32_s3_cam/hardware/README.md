# esp32_s3_cam hardware

- [`case/`](case/) — 3D-printable splash-resistant enclosures for both small
  ESP32-S3 camera boards (GOOUUU and Waveshare GC0308). Parametric build123d
  source in `case/src/`, board reference models in `case/ref/`, printable
  3MF/STL per variant in `case/export/`. See [`case/README.md`](case/README.md)
  for the interface spec, the deployed pose, and the caliper caveat.

A carrier PCB and camera-specific optics/power sizing would also live here;
neither exists yet.

Conventions follow [`../../pi5cam/hardware/`](../../pi5cam/hardware/): design
source in git, exports regenerated under `export/`. Features shared across
camera enclosures live in
[`../../hardware_common/`](../../hardware_common/).
