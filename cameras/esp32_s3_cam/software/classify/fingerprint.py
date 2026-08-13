#!/usr/bin/env python3
"""Few-shot pump-sound fingerprinter on PANNs CNN14 embeddings (2048-d).

train: embed reference wavs (/hd2/pumpaudio/refs/<label>/*.wav), build one
  L2-normalized centroid per label, calibrate per-label thresholds against
  negatives mined from the same archive (non-pump loud events + quiet floor).
  Writes /hd2/pumpaudio/fingerprint.json.

scan [date]: score every loud event's 2 s windows against the centroids and
  merge detections into /hd2/pumpaudio/labels/<date>.json as event["fp"]
  = [[label, best_sim], ...] for the UI (purple chips, searchable).

Usage: .venv/bin/python fingerprint.py train
       .venv/bin/python fingerprint.py scan [YYYY-MM-DD]
"""
import glob
import json
import os
import struct
import sys
import wave
from datetime import datetime

import numpy as np
import librosa

DATA = "/hd2/pumpaudio"
RATE = 8000
MODEL_SR = 32000
CKPT = "/hd2/models/panns/Cnn14_mAP=0.431.pth"
FP_PATH = os.path.join(DATA, "fingerprint.json")
NODATA = 0xFFFF
WIN = 2          # seconds per scoring window
THRESH_LOUD = 28
# PANNs top-labels that suggest the event may BE the pump — excluded from
# auto-mined negatives so calibration isn't poisoned
PUMPISH = {"Tick-tock", "Tick", "Clock", "Hum", "Mains hum"}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_day import load_levels, event_audio, find_events  # noqa: E402

_at = None
_masker_idx = None
# AudioSet classes that mask/confound the pump sound; their clipwise probs
# become features so the classifier learns "pump-like + loud music = doubt"
MASKERS = ["Music", "Speech", "Singing", "Musical instrument", "Radio",
           "Television", "Animal", "Walk, footsteps", "Door",
           "Computer keyboard"]


def model():
    global _at, _masker_idx
    if _at is None:
        import torch
        from panns_inference import AudioTagging, labels as as_labels
        _at = AudioTagging(checkpoint_path=CKPT,
                           device="cuda" if torch.cuda.is_available() else "cpu")
        _masker_idx = [as_labels.index(m) for m in MASKERS]
    return _at


def embed(x32k):
    """x32k: float32 mono at 32 kHz -> 2048-d CNN14 embedding + 256-d log
    power spectrum (0-4 kHz). The embedding says "motor-with-clicks"; the
    spectrum separates WHICH motor (hum line frequencies differ per machine).
    """
    want = MODEL_SR * WIN
    if len(x32k) < want:
        x32k = np.pad(x32k, (0, want - len(x32k)))
    x = x32k[:want].astype(np.float32)
    clipwise, emb = model().inference(x[None, :])
    # averaged log-PSD, 8192-pt frames -> bins 0..4 kHz, pooled to 256 dims
    n, hop = 8192, 4096
    frames = [x[i:i + n] * np.hanning(n)
              for i in range(0, len(x) - n + 1, hop)]
    psd = np.mean([np.abs(np.fft.rfft(f))[:1024] ** 2 for f in frames], axis=0)
    logpsd = np.log10(psd + 1e-10).reshape(256, 4).mean(axis=1)
    maskers = np.log(clipwise[0][_masker_idx].astype(np.float64) + 1e-4)
    return np.concatenate([emb[0].astype(np.float64), logpsd, maskers])


def fit_lr(X, y, wts, l2=1e-3, iters=4000, lr=1.0):
    """Weighted logistic regression, numpy GD + momentum, bias col appended."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    v = np.zeros_like(w)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (wts * (p - y)) / wts.sum() + l2 * w
        g[-1] -= l2 * w[-1]          # don't regularize bias
        v = 0.9 * v - lr * g
        w += v
    return w


def lr_score(v, w):
    return float(v @ w[:-1] + w[-1])


def whiten(v, mu, sd):
    """Z-score by the negative pool's stats (removes the mic/room channel
    component that makes every clip from this basement look alike), then
    L2-normalize for cosine scoring."""
    z = (v - mu) / sd
    return z / (np.linalg.norm(z) + 1e-9)


def wav_audio(path):
    with wave.open(path) as w:
        pcm = w.readframes(w.getnframes())
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return librosa.resample(x, orig_sr=RATE, target_sr=MODEL_SR)


def day_window(date, s, vals):
    """2 s window starting at wall-clock second s, resampled to 32 kHz."""
    return event_audio(date, s, s + WIN - 1, vals)


def mine_negatives(date, vals, marks):
    """Negatives: (1) every window inside user '!'-prefixed negative marks —
    confirmed-not-pump ground truth; (2) loud events that neither overlap a
    mark nor look pump-ish to PANNs; (3) quiet-floor windows."""
    taken = [(m["t0"], m["t1"]) for m in marks if m["date"] == date]
    negs = []
    for m in marks:
        if m["date"] == date and m["label"].startswith("!"):
            for s in range(m["t0"], m["t1"] - WIN + 1, WIN):
                negs.append((m["label"], s))
    try:
        lab = json.load(open(os.path.join(DATA, "labels", date + ".json")))
        events = lab["events"]
    except (OSError, ValueError):
        events = []
    for e in events:
        a, b = e["start"], e["end"]
        if any(t0 < b + 5 and a - 5 < t1 for t0, t1 in taken):
            continue
        top = {name for name, _ in e.get("top", [])}
        if top & PUMPISH or not top:
            continue
        for s in range(a, min(b + 1, a + 6), WIN):     # up to 3 windows/event
            negs.append(("ev:" + "/".join(sorted(top)[:2]), s))
    quiet = [s for s in range(0, 86400 - WIN)
             if all(vals[s + k] != NODATA and vals[s + k] < 20
                    for k in range(WIN))]
    rng = np.random.default_rng(0)
    for s in rng.choice(len(quiet), size=min(15, len(quiet)), replace=False):
        negs.append(("quiet", quiet[int(s)]))
    return negs


def train():
    refs = {}
    for path in sorted(glob.glob(os.path.join(DATA, "refs", "*", "*.wav"))):
        label = os.path.basename(os.path.dirname(path))
        refs.setdefault(label, []).append((path, embed(wav_audio(path))))
    if not refs:
        sys.exit("no reference wavs — run marks_export.py first")

    marks = json.load(open(os.path.join(DATA, "marks.json")))
    dates = sorted({m["date"] for m in marks})
    neg_embs = []
    for date in dates:
        vals = load_levels(date)
        for tag, s in mine_negatives(date, vals, marks):
            a = day_window(date, s, vals)
            if a is not None and len(a) >= MODEL_SR // 2:
                neg_embs.append((tag, date, s, embed(a)))
    print("%d negatives mined from %s" % (len(neg_embs), ", ".join(dates)))

    neg_raw = np.array([e for *_, e in neg_embs])
    mu = neg_raw.mean(axis=0)
    # shrink per-dim variance toward the global scale: ReLU embeddings have
    # many all-zero dims across negatives; a raw 1e-6 floor would explode them
    gsd = float(neg_raw.std())
    sd = np.sqrt(neg_raw.var(axis=0) + (0.3 * gsd) ** 2)
    neg_embs = [(tag, d, s, whiten(e, mu, sd)) for tag, d, s, e in neg_embs]

    out = {"model": "PANNs CNN14 embedding, negative-whitened cosine",
           "win_sec": WIN,
           "trained": datetime.now().isoformat(timespec="seconds"),
           "mu": [round(float(v), 5) for v in mu],
           "sd": [round(float(v), 5) for v in sd],
           "labels": {}}
    Xneg = np.array([e for *_, e in neg_embs])
    for label, items in sorted(refs.items()):
        embs = np.array([whiten(e, mu, sd) for _, e in items])
        # discriminative head: cosine-to-refs saturates when another machine
        # sounds alike (neg sims hit 0.94); weighted logistic regression on
        # the same whitened embeddings finds the separating direction.
        X = np.vstack([embs, Xneg])
        y = np.concatenate([np.ones(len(embs)), np.zeros(len(Xneg))])
        wts = np.concatenate([np.full(len(embs), len(Xneg) / len(embs)),
                              np.ones(len(Xneg))])
        w = fit_lr(X, y, wts)
        # leave-one-positive-out honest scores
        loo = []
        for i in range(len(embs)):
            Xi = np.vstack([np.delete(embs, i, axis=0), Xneg])
            yi = np.concatenate([np.ones(len(embs) - 1), np.zeros(len(Xneg))])
            wi = np.concatenate([np.full(len(embs) - 1,
                                         len(Xneg) / max(1, len(embs) - 1)),
                                 np.ones(len(Xneg))])
            loo.append(lr_score(embs[i], fit_lr(Xi, yi, wi)))
        for sc, (path, _) in sorted(zip(loo, items)):
            print("    ref LOO %+.2f  %s%s" % (sc, os.path.basename(path),
                  "  [NOT SEPARABLE]" if sc < 0 else ""))
        scored = sorted(((lr_score(e, w), tag, d, s)
                         for tag, d, s, e in neg_embs), reverse=True)
        neg_max = scored[0][0] if scored else -99.0
        pos_min = min(loo) if loo else 99.0
        thresh = (pos_min + neg_max) / 2 if pos_min > neg_max \
            else neg_max + 0.5
        print("  top negatives vs %s (audit: real pump cycles poison these):"
              % label)
        for sc, tag, d, s in scored[:6]:
            print("    %+.2f  %s %02d:%02d:%02d  [%s]"
                  % (sc, d, s // 3600, s % 3600 // 60, s % 60, tag))
        out["labels"][label] = {
            "w": [round(float(v), 5) for v in w],
            "thresh": round(thresh, 3), "n_refs": len(embs),
            "pos_min_loo": round(pos_min, 3), "neg_max": round(neg_max, 3)}
        print("%-10s refs=%d  pos_min(LOO)=%+.2f  neg_max=%+.2f  -> thresh %+.2f%s"
              % (label, len(embs), pos_min, neg_max, thresh,
                 "  [WEAK MARGIN]" if pos_min - neg_max < 0.5 else ""))
    with open(FP_PATH, "w") as f:
        json.dump(out, f)
    print("wrote", FP_PATH)


# detection rule, validated on 2026-08-09 ground truth: real cycles show
# either a very hot window (clicks: peaks >= +3.7) or a sustained warm run
# (hum: >= 4 windows above +2.5); music false-twins were <= 3 windows,
# peak +3.14. Tune here as more days of marks accumulate.
RUN_LO = 0.5        # segment/audit floor
PEAK_HI = 3.5       # instant accept
PEAK_MID = 2.5      # accept if sustained...
RUN_MIN_WINS = 4    # ...for at least this many 1 s-hop windows


def scan(date):
    fp = json.load(open(FP_PATH))
    names = sorted(fp["labels"])
    ws = [np.array(fp["labels"][n]["w"]) for n in names]
    mu, sd = np.array(fp["mu"]), np.array(fp["sd"])
    vals = load_levels(date)
    events = find_events(vals, THRESH_LOUD)
    lab_path = os.path.join(DATA, "labels", date + ".json")
    try:
        lab = json.load(open(lab_path))
    except (OSError, ValueError):
        lab = {"date": date, "thresh": THRESH_LOUD, "events":
               [{"start": a, "end": b, "top": [], "segments": []}
                for a, b in events]}
    by_start = {e["start"]: e for e in lab["events"]}
    n_hits = 0
    for a, b in events:
        best = np.zeros(len(names))
        wins = []
        for s in range(max(0, a - 1), b + 1):
            w = day_window(date, s, vals)
            if w is None or len(w) < MODEL_SR // 2:
                continue
            wv = whiten(embed(w), mu, sd)
            sims = np.array([lr_score(wv, wl) for wl in ws])
            wins.append((s, sims))
            best = np.maximum(best, sims)
        # contiguous warm runs (score >= RUN_LO): audit trail for the UI, and
        # the substrate for the click-or-sustained detection rule
        segs = []
        detected = [False] * len(names)
        for i, nm in enumerate(names):
            run = None      # [start, last, peak, n_windows]
            for s, sims in list(wins) + [(10 ** 9, np.full(len(names), -99))]:
                if sims[i] >= RUN_LO and s < 10 ** 9:
                    if run and s - run[1] <= 2:
                        run[1], run[2], run[3] = \
                            s, max(run[2], float(sims[i])), run[3] + 1
                    else:
                        run = [s, s, float(sims[i]), 1]
                        segs.append(run)  # mutated in place as run extends
                else:
                    if run and (run[2] >= PEAK_HI or
                                (run[2] >= PEAK_MID and run[3] >= RUN_MIN_WINS)):
                        detected[i] = True
                    run = None
        segs = [[names[0], r[0], r[1] + WIN, round(r[2], 3)]
                for r in sorted(segs, key=lambda x: -x[2])[:8]]
        segs.sort(key=lambda x: x[1])
        hits = [[names[i], round(float(best[i]), 3)]
                for i in range(len(names)) if detected[i]]
        hits.sort(key=lambda h: -h[1])
        ev = by_start.get(a)
        if ev is None:
            ev = {"start": a, "end": b, "top": [], "segments": []}
            lab["events"].append(ev)
        ev["fp"] = hits
        ev["fpseg"] = segs
        if hits:
            n_hits += 1
            print("%02d:%02d:%02d-%02d:%02d:%02d  %s"
                  % (a // 3600, a % 3600 // 60, a % 60, b // 3600,
                     b % 3600 // 60, b % 60,
                     ", ".join("%s %.2f" % (n, s) for n, s in hits)))
    lab["events"].sort(key=lambda e: e["start"])
    with open(lab_path, "w") as f:
        json.dump(lab, f)
    print("%d/%d events matched a fingerprint; merged into %s"
          % (n_hits, len(events), lab_path))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "scan":
        scan(sys.argv[2] if len(sys.argv) > 2
             else datetime.now().strftime("%Y-%m-%d"))
    else:
        sys.exit(__doc__)
