#!/usr/bin/env python3
"""Export reference marks (/hd2/pumpaudio/marks.json) to per-label wav files
for fingerprint training: /hd2/pumpaudio/refs/<label>/<id>_<date>_<hhmmss>.wav
Re-runnable; skips files that already exist. Stdlib only.
"""
import json
import os
import struct

DATA = "/hd2/pumpaudio"
RATE = 8000
SEC_BYTES = RATE * 2
MINUTE_BYTES = 60 * SEC_BYTES
NODATA = 0xFFFF


def wav_header(nbytes):
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + nbytes, b"WAVE",
                       b"fmt ", 16, 1, 1, RATE, RATE * 2, 2, 16,
                       b"data", nbytes)


def second_pcm(date, s, vals, fcache):
    mkey = s // 60
    if mkey not in fcache:
        p = os.path.join(DATA, "audio", date,
                         "%02d-%02d.pcm" % (mkey // 60, mkey % 60))
        fcache[mkey] = open(p, "rb").read() if os.path.exists(p) else None
    blob = fcache[mkey]
    if blob is None or vals[s] == NODATA:
        return None
    if len(blob) == MINUTE_BYTES:
        off = (s % 60) * SEC_BYTES
        return blob[off:off + SEC_BYTES]
    present = [k for k in range(60) if vals[mkey * 60 + k] != NODATA]
    try:
        idx = present.index(s % 60)
    except ValueError:
        return None
    if (idx + 1) * SEC_BYTES <= len(blob):
        return blob[idx * SEC_BYTES:(idx + 1) * SEC_BYTES]
    return None


def main():
    marks = json.load(open(os.path.join(DATA, "marks.json")))
    lv, fcaches = {}, {}
    n_new = 0
    for m in marks:
        if m["label"].startswith("!"):     # negative marks: train-time only
            continue
        d = m["date"]
        if d not in lv:
            with open(os.path.join(DATA, "levels", d + ".u16"), "rb") as f:
                lv[d] = struct.unpack("<86400H", f.read())
            fcaches[d] = {}
        label = "".join(c if c.isalnum() or c in "-_" else "_"
                        for c in m["label"])
        outdir = os.path.join(DATA, "refs", label)
        os.makedirs(outdir, exist_ok=True)
        t0 = m["t0"]
        name = "%03d_%s_%02d%02d%02d.wav" % (
            m["id"], d, t0 // 3600, t0 % 3600 // 60, t0 % 60)
        outp = os.path.join(outdir, name)
        if os.path.exists(outp):
            continue
        chunks = [second_pcm(d, s, lv[d], fcaches[d])
                  for s in range(m["t0"], m["t1"])]
        pcm = b"".join(c for c in chunks if c)
        if not pcm:
            print("SKIP (no audio):", m)
            continue
        with open(outp, "wb") as f:
            f.write(wav_header(len(pcm)))
            f.write(pcm)
        n_new += 1
        print("wrote %s (%.1fs)" % (outp, len(pcm) / SEC_BYTES))
    print("%d new files" % n_new)


if __name__ == "__main__":
    main()
