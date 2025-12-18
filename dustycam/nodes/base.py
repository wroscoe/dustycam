from __future__ import annotations

import platform

def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    return machine.startswith("arm") or machine.startswith("aarch64")