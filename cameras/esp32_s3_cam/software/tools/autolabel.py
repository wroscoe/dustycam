#!/usr/bin/env python3
"""Auto-label sample frames with a stronger vision model (teacher-student).

Runs YOLO (COCO-pretrained) over every dataset/samples/*.png and writes
dataset/labels.csv with: file, person (0/1), teacher_conf, tiny_model_score.
Meant to run inside a container: see `make autolabel`.
"""
import csv
import glob
import json
import os
import sys

from PIL import Image
from ultralytics import YOLO

# Dataset lives outside the repo (bulk storage). Resolution order:
# DATASET_ROOT env var (set by the containerized make targets), then
# ~/.dusty/config.toml [paths] dataset_root, then this default.
def _dataset_root():
    env = os.environ.get('DATASET_ROOT')
    if env:
        return env
    try:
        from dusty.config import load
        return load(required=False).get('paths', {}).get('dataset_root') \
            or '/hd2/datasets/wavesharecam'
    except Exception:
        return '/hd2/datasets/wavesharecam'


DATASET_ROOT = _dataset_root()

SAMPLES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATASET_ROOT, 'samples')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATASET_ROOT, 'labels.csv')
CONF_THRESHOLD = 0.30
PERSON_CLASS = 0          # COCO class 0 = person
UPSCALE = 384             # 96 -> 384 so YOLO has enough pixels


def main():
    model = YOLO('yolov8s.pt')
    pngs = sorted(glob.glob(os.path.join(SAMPLES, '*.png')))
    print(f'labeling {len(pngs)} frames...', flush=True)
    rows = []
    batch, names = [], []
    for i, p in enumerate(pngs):
        img = Image.open(p).convert('RGB').resize((UPSCALE, UPSCALE),
                                                  Image.LANCZOS)
        batch.append(img)
        names.append(p)
        if len(batch) == 64 or i == len(pngs) - 1:
            results = model.predict(batch, conf=CONF_THRESHOLD,
                                    classes=[PERSON_CLASS], verbose=False)
            for name, res in zip(names, results):
                confs = [float(b.conf) for b in res.boxes]
                person = 1 if confs else 0
                tiny = -1.0
                j = name[:-4] + '.json'
                if os.path.exists(j):
                    tiny = json.load(open(j)).get('person_score', -1.0)
                rows.append((os.path.basename(name), person,
                             max(confs) if confs else 0.0, tiny))
            batch, names = [], []
            print(f'  {i+1}/{len(pngs)}', flush=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'person', 'teacher_conf', 'tiny_score'])
        w.writerows(rows)
    n_person = sum(r[1] for r in rows)
    print(f'done: {n_person} person / {len(rows) - n_person} no-person '
          f'-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
