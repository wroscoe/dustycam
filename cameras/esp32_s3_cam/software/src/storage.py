"""Image storage: internal flash filesystem (SD needs custom firmware).

Layout:
    /imgs/pending/NNNNNN.jpg   captured, not yet uploaded
    /imgs/sent/                uploaded (kept until space is needed)
    /imgs/seq.txt              next sequence number
"""
import os

ROOT = '/imgs'
PENDING = ROOT + '/pending'
SENT = ROOT + '/sent'
_SEQ = ROOT + '/seq.txt'

MIN_FREE_KB = 512     # stop capturing below this much free flash


def _mkdirs():
    for d in (ROOT, PENDING, SENT):
        try:
            os.mkdir(d)
        except OSError:
            pass


def init():
    _mkdirs()


def free_kb():
    st = os.statvfs('/')
    return st[0] * st[3] // 1024


def next_seq():
    try:
        with open(_SEQ) as f:
            n = int(f.read())
    except (OSError, ValueError):
        n = 0
    with open(_SEQ, 'w') as f:
        f.write(str(n + 1))
    return n


def save_jpeg(data):
    """Save a JPEG; returns path or None if out of space."""
    if free_kb() < MIN_FREE_KB:
        _reclaim()
        if free_kb() < MIN_FREE_KB:
            return None
    path = '%s/%06d.jpg' % (PENDING, next_seq())
    with open(path, 'wb') as f:
        f.write(data)
    return path


def pending():
    try:
        return sorted(PENDING + '/' + f for f in os.listdir(PENDING))
    except OSError:
        return []


def mark_sent(path):
    name = path.rsplit('/', 1)[1]
    os.rename(path, SENT + '/' + name)


def _reclaim():
    """Delete oldest already-sent images to make room."""
    try:
        sent = sorted(os.listdir(SENT))
    except OSError:
        return
    for name in sent[:20]:
        try:
            os.remove(SENT + '/' + name)
        except OSError:
            pass


def log(msg):
    """Append to an on-flash log (console prints wedge when no host reads)."""
    try:
        with open(ROOT + '/log.txt', 'a') as f:
            f.write(msg + '\n')
    except OSError:
        pass
