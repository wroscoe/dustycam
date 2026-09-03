"""config: tuning = defaults stamped into the firmware, overridden by the
last server config pulled (`GET /config/<device>`) and kept on flash.
Standard §5. Keys are only accepted if they exist in the defaults, and are
coerced to the default's type; `cfg` is the server's version number.
"""
import json

import secrets

from uplink import *

CFG = {}                      # live tuning; modules read CFG[...] at call time
CFG_FILE = '/flash/config.json'


def _coerce(default, value):
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)


def cfg_apply(obj, defaults):
    """Merge a server config object into CFG. Returns the list of keys that
    changed. Unknown keys are ignored (firmware first, then config)."""
    changed = []
    for k, v in obj.items():
        if k == 'cfg':
            nv = int(v)
        elif k == 'mode':
            nv = str(v)
        elif k in defaults:
            try:
                nv = _coerce(defaults[k], v)
            except (TypeError, ValueError):
                continue
        else:
            continue
        if CFG.get(k) != nv:
            CFG[k] = nv
            changed.append(k)
    return changed


def cfg_init(defaults):
    """Defaults, then whatever the last pull stored."""
    CFG.clear()
    CFG.update(defaults)
    CFG['cfg'] = 0
    CFG['mode'] = 'live'
    try:
        with open(CFG_FILE) as f:
            cfg_apply(json.loads(f.read()), defaults)
    except (OSError, ValueError):
        pass
    return CFG


def cfg_pull(defaults):
    """Fetch the server config; apply and store if its version differs.
    Returns the changed keys (empty if nothing new or unreachable)."""
    status, body = http_get('/config/%s' % secrets.DEVICE, 4096, 10)
    if status != 200 or not body:
        return []
    try:
        obj = json.loads(body.decode() if isinstance(body, bytes) else body)
    except ValueError:
        print('config: bad json')
        return []
    if int(obj.get('cfg', 0)) == CFG.get('cfg'):
        return []
    changed = cfg_apply(obj, defaults)
    try:
        with open(CFG_FILE, 'w') as f:
            f.write(body if isinstance(body, str) else body.decode())
    except OSError as e:
        print('config: store failed', repr(e))
    print('config: cfg %s applied, changed %s' % (CFG.get('cfg'), changed))
    return changed
