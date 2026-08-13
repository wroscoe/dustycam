#!/usr/bin/env python3
"""Classify a day's loud events with PANNs CNN14 (AudioSet, 527 classes).

Finds loud runs in the day's 1 Hz level log (same threshold logic as the UI),
assembles each event's audio from the per-minute .pcm files, and runs:
  - AudioTagging (clipwise): top labels per event
  - SoundEventDetection (framewise ~0.32 s): labeled segments within events

Writes /hd2/pumpaudio/labels/<date>.json for the UI.
Usage: .venv/bin/python classify_day.py [YYYY-MM-DD] [--thresh N]
"""
import json
import os
import struct
import sys
from datetime import datetime

import numpy as np
import librosa
import torch
from panns_inference import AudioTagging, SoundEventDetection, labels

DATA = "/hd2/pumpaudio"
RATE = 8000
MODEL_SR = 32000
CKPT_DIR = "/hd2/models/panns"
NODATA = 0xFFFF
MERGE_GAP = 90
MIN_SEC = 3
PAD = 5          # context seconds around each event


def load_levels(date):
    with open(os.path.join(DATA, "levels", date + ".u16"), "rb") as f:
        return struct.unpack("<86400H", f.read())


def find_events(vals, thresh):
    runs, start, last = [], None, None
    for i, v in enumerate(vals):
        if v != NODATA and v >= thresh:
            if start is None or i - last > MERGE_GAP:
                if start is not None:
                    runs.append((start, last))
                start = i
            last = i
    if start is not None:
        runs.append((start, last))
    return [(a, b) for a, b in runs if b - a + 1 >= MIN_SEC]


SEC_BYTES = RATE * 2
MINUTE_BYTES = 60 * SEC_BYTES


def event_audio(date, a, b, vals):
    """Extract exactly [a-PAD, b+PAD] worth of *present* audio seconds, so
    short sounds aren't diluted by minutes of silence. Handles both aligned
    (960000 B) and legacy compact minute files via the level presence map."""
    chunks = []
    fcache = {}
    for s in range(max(0, a - PAD), min(86399, b + PAD) + 1):
        if vals[s] == NODATA:
            continue
        mkey = s // 60
        if mkey not in fcache:
            p = os.path.join(DATA, "audio", date,
                             "%02d-%02d.pcm" % (mkey // 60, mkey % 60))
            fcache[mkey] = open(p, "rb").read() if os.path.exists(p) else None
        blob = fcache[mkey]
        if blob is None:
            continue
        if len(blob) == MINUTE_BYTES:
            off = (s % 60) * SEC_BYTES
            chunks.append(blob[off:off + SEC_BYTES])
        else:
            present = [k for k in range(60) if vals[mkey * 60 + k] != NODATA]
            try:
                idx = present.index(s % 60)
                if (idx + 1) * SEC_BYTES <= len(blob):
                    chunks.append(blob[idx * SEC_BYTES:(idx + 1) * SEC_BYTES])
            except ValueError:
                pass
    if not chunks:
        return None
    pcm = b"".join(chunks)
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return librosa.resample(x, orig_sr=RATE, target_sr=MODEL_SR)


def fmt(s):
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def main():
    date = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else datetime.now().strftime("%Y-%m-%d")
    thresh = 28
    if "--thresh" in sys.argv:
        thresh = int(sys.argv[sys.argv.index("--thresh") + 1])

    vals = load_levels(date)
    events = find_events(vals, thresh)
    print("date %s, %d events at thresh %d" % (date, len(events), thresh))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)
    at = AudioTagging(checkpoint_path=os.path.join(CKPT_DIR, "Cnn14_mAP=0.431.pth"),
                      device=device)
    sed = SoundEventDetection(
        checkpoint_path=os.path.join(CKPT_DIR, "Cnn14_DecisionLevelMax_mAP=0.385.pth"),
        device=device)

    out = {"date": date, "thresh": thresh, "model": "PANNs CNN14 (AudioSet 527)",
           "events": []}
    for a, b in events:
        audio = event_audio(date, a, b, vals)
        if audio is None or len(audio) < MODEL_SR:
            out["events"].append({"start": a, "end": b, "top": [], "segments": [],
                                  "note": "no audio"})
            continue
        clip = audio[None, :]
        clipwise, _ = at.inference(clip)
        top_idx = np.argsort(clipwise[0])[::-1][:6]
        top = [[labels[i], round(float(clipwise[0][i]), 3)] for i in top_idx
               if clipwise[0][i] >= 0.05]

        framewise = sed.inference(clip)[0]           # (frames, 527)
        hop = 0.032 * 10                             # 320 ms per frame
        segments = []
        for ci in {int(i) for i in top_idx[:4]}:
            probs = framewise[:, ci]
            on = probs >= max(0.25, 0.5 * probs.max())
            i = 0
            while i < len(on):
                if on[i]:
                    j = i
                    while j < len(on) and on[j]:
                        j += 1
                    if (j - i) * hop >= 0.5:
                        segments.append({
                            "t0": round(a - PAD + i * hop, 1),
                            "t1": round(a - PAD + j * hop, 1),
                            "label": labels[ci],
                            "p": round(float(probs[i:j].max()), 3)})
                    i = j
                else:
                    i += 1
        segments.sort(key=lambda s: s["t0"])
        out["events"].append({"start": a, "end": b, "top": top,
                              "segments": segments})
        print("%s -> %s  %s" % (fmt(a), fmt(b),
              ", ".join("%s %.2f" % (l, p) for l, p in top[:4])))

    os.makedirs(os.path.join(DATA, "labels"), exist_ok=True)
    with open(os.path.join(DATA, "labels", date + ".json"), "w") as f:
        json.dump(out, f)
    print("wrote", os.path.join(DATA, "labels", date + ".json"))


if __name__ == "__main__":
    main()
