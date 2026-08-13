"""Load DustyCam configuration and secrets from ``~/.dusty/``.

Two files, merged into one view:

* ``config.toml``  — non-secret settings, safe to read anywhere
* ``secrets.toml`` — credentials; must be mode 0600

Both may carry ``[camera.<name>]`` sections. :func:`load` flattens the one
matching the requested camera into ``cfg.camera``, so callers never index by
name themselves.

    >>> cfg = load(camera="openmv_n6")
    >>> cfg.wifi.ssid, cfg.camera.device
    ('...', 'n6cam')

Values from ``secrets.toml`` win on conflict. Environment variables of the
form ``DUSTY_<SECTION>_<KEY>`` win over both, which is how CI and one-off
overrides stay out of the files.
"""

from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path

DUSTY_HOME = Path(os.environ.get("DUSTY_HOME", Path.home() / ".dusty"))

CONFIG_FILE = "config.toml"
SECRETS_FILE = "secrets.toml"


class ConfigError(RuntimeError):
    """Raised when ~/.dusty is missing, malformed, or unsafely permissioned."""


class Section(dict):
    """A dict whose keys are also attributes, so ``cfg.wifi.ssid`` reads well."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"no setting {name!r} (have: {', '.join(sorted(self)) or 'nothing'})"
            ) from None

    def __setattr__(self, name, value):
        self[name] = value


def _wrap(value):
    if isinstance(value, dict):
        return Section({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge; overlay wins on scalar conflicts."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_toml(path: Path, *, require_private: bool = False) -> dict:
    if not path.exists():
        return {}
    if require_private:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ConfigError(
                f"{path} is readable by others (mode {mode:04o}). "
                f"Fix with: chmod 600 {path}"
            )
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc


def _env_overrides(data: dict) -> dict:
    """Apply DUSTY_<SECTION>_<KEY> environment variables over the loaded data."""
    for var, value in os.environ.items():
        if not var.startswith("DUSTY_") or var == "DUSTY_HOME":
            continue
        parts = var[len("DUSTY_"):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        if section in data and isinstance(data[section], dict):
            data[section][key] = value
    return data


def load(camera: str | None = None, *, required: bool = True) -> Section:
    """Return the merged configuration, optionally scoped to one camera.

    With ``camera`` given, that camera's ``[camera.<name>]`` section is
    promoted to ``cfg.camera`` and the per-name mapping is dropped.

    Pass ``required=False`` to get an empty Section instead of an error when
    ``~/.dusty/`` does not exist yet — useful for callers that have their own
    fallback (an existing environment variable, say).
    """
    if not DUSTY_HOME.exists():
        if not required:
            return Section()
        raise ConfigError(
            f"{DUSTY_HOME} does not exist. Create it with:  dusty init"
        )

    data = _merge(
        _read_toml(DUSTY_HOME / CONFIG_FILE),
        _read_toml(DUSTY_HOME / SECRETS_FILE, require_private=True),
    )
    data = _env_overrides(data)

    cameras = data.pop("camera", {}) or {}
    if camera is not None:
        if camera not in cameras:
            known = ", ".join(sorted(cameras)) or "none defined"
            raise ConfigError(
                f"no [camera.{camera}] section in {DUSTY_HOME} (have: {known})"
            )
        data["camera"] = cameras[camera]
        data["camera_name"] = camera

    return _wrap(data)
