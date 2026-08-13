#!/usr/bin/env python3
"""Fine-tune the person detector on auto-labeled camera samples.

Teacher-student: labels come from dataset/labels.csv (YOLO teacher).
Architecture mirrors the stock TFLM person_detect model: MobileNetV1-ish
alpha=0.25, 96x96x1 input, 2-class softmax — same op set (Conv2D,
DepthwiseConv2D, AveragePool2D, Reshape, Softmax) so the firmware's op
resolver needs no changes.

Outputs:
  dataset/model/person_detect.tflite        (int8 quantized)
  dataset/model/person_detect_model_data.cc (drop-in for persondet_app)

Run in a container: see `make train`.
"""
import csv
import os
import random
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

import os
# Dataset lives outside the repo (bulk storage). Resolution order:
# Override with the DATASET_ROOT env var (the containerized make
# targets set it).
DATASET_ROOT = os.environ.get('DATASET_ROOT', '/hd2/datasets/wavesharecam')
SAMPLES = os.path.join(DATASET_ROOT, 'samples')
LABELS = os.path.join(DATASET_ROOT, 'labels.csv')
OUTDIR = os.path.join(DATASET_ROOT, 'model')
IMG = 96
EPOCHS = int(os.environ.get('EPOCHS', '30'))
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    rows = list(csv.DictReader(open(LABELS)))
    random.shuffle(rows)
    xs, ys = [], []
    for r in rows:
        p = os.path.join(SAMPLES, r['file'])
        if not os.path.exists(p):
            continue
        img = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
        xs.append(img[..., None] / 127.5 - 1.0)     # [-1, 1]
        ys.append(int(r['person']))
    x = np.stack(xs)
    y = np.array(ys, dtype=np.int32)
    n_val = max(200, len(x) // 10)
    return (x[n_val:], y[n_val:]), (x[:n_val], y[:n_val])


def depthwise_block(x, filters, stride):
    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same',
                                        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.Conv2D(filters, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU(6.)(x)


def build_model(alpha=0.25):
    def c(n):
        return max(8, int(n * alpha))
    inp = tf.keras.Input((IMG, IMG, 1))
    x = tf.keras.layers.Conv2D(c(32), 3, strides=2, padding='same',
                               use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    for filters, stride in [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
                            (512, 2), (512, 1), (512, 1), (512, 1), (512, 1),
                            (512, 1), (1024, 2), (1024, 1)]:
        x = depthwise_block(x, c(filters), stride)
    x = tf.keras.layers.AveragePooling2D(3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2)(x)
    out = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inp, out)


def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.3)
    x = tf.image.random_contrast(x, 0.6, 1.4)
    # random shifts up to 8px
    x = tf.image.resize_with_crop_or_pad(x, IMG + 16, IMG + 16)
    x = tf.image.random_crop(x, (IMG, IMG, 1))
    return tf.clip_by_value(x, -1., 1.), y


def main():
    (xt, yt), (xv, yv) = load_data()
    n_pos = int(yt.sum())
    print(f'train {len(xt)} ({n_pos} person), val {len(xv)}', flush=True)
    if n_pos < 50 or n_pos > len(yt) - 50:
        print('WARNING: classes very imbalanced — collect more varied data')

    ds = (tf.data.Dataset.from_tensor_slices((xt, yt))
          .shuffle(4096, seed=SEED).map(augment).batch(64)
          .prefetch(tf.data.AUTOTUNE))
    vds = tf.data.Dataset.from_tensor_slices((xv, yv)).batch(64)

    model = build_model()
    # class weights balance the (likely) skewed label distribution
    w1 = len(yt) / (2 * max(n_pos, 1))
    w0 = len(yt) / (2 * max(len(yt) - n_pos, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(ds, validation_data=vds, epochs=EPOCHS,
              class_weight={0: w0, 1: w1},
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=6, restore_best_weights=True)],
              verbose=2)

    loss, acc = model.evaluate(vds, verbose=0)
    print(f'val accuracy: {acc:.3f}', flush=True)

    # ---- int8 quantization ----
    def rep_data():
        for i in range(0, min(500, len(xt))):
            yield [xt[i:i+1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()

    os.makedirs(OUTDIR, exist_ok=True)
    open(f'{OUTDIR}/person_detect.tflite', 'wb').write(tfl)
    print(f'tflite size: {len(tfl)} bytes', flush=True)

    # ---- emit C array (drop-in replacement) ----
    lines = ['#include "person_detect_model_data.h"', '',
             'alignas(16) const unsigned char g_person_detect_model_data[] = {']
    for i in range(0, len(tfl), 12):
        chunk = ', '.join(f'0x{b:02x}' for b in tfl[i:i+12])
        lines.append('    ' + chunk + ',')
    lines.append('};')
    lines.append(f'const int g_person_detect_model_data_len = {len(tfl)};')
    open(f'{OUTDIR}/person_detect_model_data.cc', 'w').write('\n'.join(lines))
    print('wrote model .cc — copy into persondet_app/main/ and ota-deploy',
          flush=True)


if __name__ == '__main__':
    main()
