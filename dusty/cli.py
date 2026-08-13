"""``dusty`` command line: manage ~/.dusty/ and generate board credentials."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

from .config import CONFIG_FILE, DUSTY_HOME, SECRETS_FILE, ConfigError, load
from .generate import TARGETS, generate, write_private

CONFIG_TEMPLATE = """\
# DustyCam settings. Non-secret — credentials belong in secrets.toml.

[server]
host = "192.168.1.100"   # this workstation's LAN IP, as the boards see it
port = 8077

[paths]
dataset_root = "/hd2/datasets/wavesharecam"

[camera.esp32_s3_cam]
device = "wavecam"

[camera.openmv_n6]
device = "n6cam"
period_s = 10            # seconds between change checks
diff_min_frac = 0.005    # upload if >0.5% of pixels changed
diff_l_thresh = 8        # per-pixel lightness delta (of 100) counted as changed
heartbeat_s = 300        # force an upload at least this often
ota_port = 8266
"""

SECRETS_TEMPLATE = """\
# DustyCam credentials. Mode 0600; never committed.
# Anything here overrides the same key in config.toml.

[wifi]
ssid = "your-ssid"
password = "your-password"

[mqtt]
user = ""
password = ""
topic = "sensorhub"

[google]
api_key = ""             # used by `dustycam make` / image generation

[camera.openmv_n6]
ota_token = ""           # shared secret for ./ota_push.py
"""


def repo_root() -> Path:
    """Locate the repo root (holds pyproject.toml), from source or cwd."""
    here = Path(__file__).resolve().parent.parent
    if (here / "pyproject.toml").exists():
        return here
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise ConfigError("cannot locate the repo root (no pyproject.toml found)")


def cmd_init(args) -> int:
    DUSTY_HOME.mkdir(parents=True, exist_ok=True)
    os.chmod(DUSTY_HOME, stat.S_IRWXU)  # 0700

    created = []
    config_path = DUSTY_HOME / CONFIG_FILE
    if config_path.exists() and not args.force:
        print(f"  exists  {config_path}")
    else:
        config_path.write_text(CONFIG_TEMPLATE)
        created.append(config_path)

    secrets_path = DUSTY_HOME / SECRETS_FILE
    if secrets_path.exists() and not args.force:
        print(f"  exists  {secrets_path}")
    else:
        write_private(secrets_path, SECRETS_TEMPLATE)
        created.append(secrets_path)

    for path in created:
        print(f"  created {path}")
    if created:
        print(f"\nEdit {secrets_path}, then run:  dusty generate --all")
    return 0


def _redact(value) -> str:
    """Report only whether a credential is set, and how long — never a prefix."""
    text = str(value)
    if not text:
        return "(empty)"
    return f"(set, {len(text)} chars)"


SECRET_HINTS = ("password", "pass", "token", "api_key", "secret")


def cmd_show(args) -> int:
    cfg = load(camera=args.camera)
    print(f"# {DUSTY_HOME}  (secrets redacted)")
    for section, body in sorted(cfg.items()):
        if not isinstance(body, dict):
            print(f"{section} = {body!r}")
            continue
        print(f"\n[{section}]")
        for key, value in sorted(body.items()):
            shown = _redact(value) if any(h in key for h in SECRET_HINTS) else repr(value)
            print(f"{key} = {shown}")
    return 0


def cmd_generate(args) -> int:
    cameras = sorted(TARGETS) if args.all else [args.camera]
    if not cameras or cameras == [None]:
        print("specify a camera, or --all", file=sys.stderr)
        return 2
    root = repo_root()
    for camera in cameras:
        for path in generate(camera, root):
            print(f"  wrote {path.relative_to(root)}  (0600)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dusty", description="Manage ~/.dusty/ configuration and secrets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help=f"create {DUSTY_HOME} from templates")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.set_defaults(func=cmd_init)

    p_show = sub.add_parser("show", help="print the merged config, secrets redacted")
    p_show.add_argument("--camera", help="scope to one camera")
    p_show.set_defaults(func=cmd_show)

    p_gen = sub.add_parser("generate", help="write board-side secrets files")
    p_gen.add_argument("camera", nargs="?", choices=sorted(TARGETS))
    p_gen.add_argument("--all", action="store_true", help="every camera")
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
