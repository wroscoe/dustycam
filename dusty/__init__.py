"""Shared host-side library for DustyCam.

Configuration and credentials live outside the repo, in ``~/.dusty/``:

    ~/.dusty/config.toml    non-secret settings (checked into nothing)
    ~/.dusty/secrets.toml   credentials, mode 0600

Nothing here is importable on a microcontroller — board-side ``secrets.py``
is *generated* from the above by :mod:`dusty.generate` at deploy time.
"""

from .config import DUSTY_HOME, ConfigError, load

__all__ = ["load", "DUSTY_HOME", "ConfigError"]
