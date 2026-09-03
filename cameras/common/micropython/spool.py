"""spool: offline buffer on the SD card. Standard §2 stage 6 (Record/Deliver).

Sidecar JSON first, then the image to a temp name, then rename: a .jpg on
disk always has its .json. Drain oldest-first with recording paused.
"""
import os
import time

from uplink import *

PENDING_DIR = '/sdcard/pending'


def sd_ready():
    """True if a card is mounted and the pending dir exists (created if not)."""
    try:
        os.stat('/sdcard')
    except OSError:
        return False
    try:
        os.mkdir(PENDING_DIR)
    except OSError:
        pass
    try:
        os.listdir(PENDING_DIR)
        return True
    except OSError:
        return False


def pending_count():
    try:
        return sum(1 for f in os.listdir(PENDING_DIR) if f.endswith('.jpg'))
    except OSError:
        return 0


def spool_name(ts, exists):
    """Base path for a frame captured at epoch `ts`; `exists(path)` says
    whether a .jpg is already there (pure, for tests)."""
    base = '%s/%010d' % (PENDING_DIR, ts)
    name, i = base, 0
    while exists(name + '.jpg'):
        i += 1
        name = '%s_%d' % (base, i)
    return name


def _exists(path):
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def buffer_frame(data, meta):
    """Write meta sidecar + jpg. True on success."""
    name = spool_name(time.time(), _exists)
    try:
        with open(name + '.json', 'w') as f:
            f.write(meta)
        with open(name + '.tmp', 'wb') as f:
            f.write(data)
        os.rename(name + '.tmp', name + '.jpg')
        return True
    except OSError:
        for ext in ('.json', '.tmp'):
            try:
                os.remove(name + ext)
            except OSError:
                pass
        return False


def drain_pending(idle, version):
    """Upload the whole backlog oldest-first, streaming each file. `idle()`
    is called between files (keep the control plane responsive). Returns
    True when the dir is empty."""
    try:
        files = sorted(f for f in os.listdir(PENDING_DIR) if f.endswith('.jpg'))
    except OSError:
        return True
    for f in files:
        base = PENDING_DIR + '/' + f[:-4]
        try:
            try:
                with open(base + '.json') as fp:
                    meta = fp.read()
            except OSError:
                meta = '{"buffered": true, "v": "%s"}' % version
            with open(base + '.jpg', 'rb') as fp:
                ok = post_blob('frame', fp, meta, os.stat(base + '.jpg')[6])
        except OSError:
            continue
        if not ok:
            return False                    # server gone again; resume offline
        for ext in ('.jpg', '.json'):
            try:
                os.remove(base + ext)
            except OSError:
                pass
        idle()
    return True
