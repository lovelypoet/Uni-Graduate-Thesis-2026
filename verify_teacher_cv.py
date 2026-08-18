#!/usr/bin/env python3
"""
verify_cv.py — Chạy CV nhiều lần với SEED khác nhau để kiểm chứng độ ổn định.

Mục đích: v1 có std cao (0.089) — do data thật sự bất ổn, hay chỉ 1 fold xui?
Chạy CV với 3 seed khác nhau → nếu std vẫn cao → bất ổn thật.

Chạy: source setup_gpu.sh && python3 verify_cv.py
(Lưu ý: chạy trên DATASET_DIR hiện tại trong config — đặt = "dataset" cho v1)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import config as C

N_FOLDS = 5
CV_EPOCHS_HEAD = 20
CV_EPOCHS_FT   = 15
SEEDS = [42, 123, 2024]   # 3 seed khác nhau để verify


def gather_all():
    paths, labels = [], []
    for split in ["train", "val", "test"]:
        for idx, cls in enumerate(C.CLASS_NAMES):
            d = C.DATASET_DIR / split / cls
            if not d.exists():
                continue
            for f in d.glob("*.jpg"):
                paths.append(str(f)); labels.append(idx)
    return np.array(paths), np.array(labels)


def _decode(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
    return img, label


def make_ds(paths, labels, training, seed):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(512, seed=seed)
        if getattr(C, "USE_AUGMENTATION", True):
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomBrightness(0.10),
                tf.keras.layers.RandomContrast(0.10)])
            ds = ds.map(lambda i, l: (aug(i, training=True), l),
                        num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build():
    inp = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inp)
    bb = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x)
    bb.trainable = False
    x = layers.GlobalAveragePooling2D()(bb.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return models.Model(inp, layers.Dense(C.NUM_CLASSES)(x)), bb


def cw_from(labels):
    total = len(labels)
    return {i: (total/(C.NUM_CLASSES*np.sum(labels==i))
                if np.sum(labels==i)>0 else 1.0) for i in range(C.NUM_CLASSES)}


def run_cv(paths, labels, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accs, f1s = [], []
    for trainval_idx, te_idx in skf.split(paths, labels):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        tr_sub, val_sub = next(inner.split(paths[trainval_idx], labels[trainval_idx]))
        tr_idx, val_idx = trainval_idx[tr_sub], trainval_idx[val_sub]
        tr = make_ds(paths[tr_idx], labels[tr_idx], True, seed)
        va = make_ds(paths[val_idx], labels[val_idx], False, seed)
        te = make_ds(paths[te_idx], labels[te_idx], False, seed)
        cw = cw_from(labels[tr_idx])
        m, bb = build()
        es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True,
                                              monitor="val_accuracy", mode="max")
        m.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                  loss=loss_fn, metrics=["accuracy"])
        m.fit(tr, validation_data=va, epochs=CV_EPOCHS_HEAD,
              class_weight=cw, callbacks=[es], verbose=0)
        bb.trainable = True
        for l in bb.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False
        m.compile(optimizer=tf.keras.optimizers.Adam(C.FINE_TUNE_LR),
                  loss=loss_fn, metrics=["accuracy"])
        m.fit(tr, validation_data=va, epochs=CV_EPOCHS_FT,
              class_weight=cw, callbacks=[es], verbose=0)
        yt, yp = [], []
        for x, y in te:
            yp.extend(tf.argmax(m(x, training=False), 1).numpy()); yt.extend(y.numpy())
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        tf.keras.backend.clear_session()
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def main():
    print(f"VERIFY CV — dataset: {C.DATASET_DIR}, aug: {getattr(C,'USE_AUGMENTATION',True)}")
    paths, labels = gather_all()
    print(f"Tổng ảnh: {len(paths)}\n")

    all_acc = []
    for seed in SEEDS:
        print(f"=== SEED {seed} ===")
        am, ast_, fm, fst = run_cv(paths, labels, seed)
        print(f"  Accuracy: {am:.4f} ± {ast_:.4f} | F1: {fm:.4f} ± {fst:.4f}")
        all_acc.append(am)

    print(f"\n{'='*50}")
    print(f"TỔNG HỢP {len(SEEDS)} seed:")
    print(f"  Accuracy trung bình các seed: {np.mean(all_acc):.4f}")
    print(f"  Dao động giữa các seed (std): {np.std(all_acc):.4f}")
    print(f"{'='*50}")
    if np.std(all_acc) < 0.02:
        print("→ Ổn định qua các seed → con số đáng tin")
    else:
        print("→ Dao động lớn giữa seed → data thật sự bất ổn (nhỏ)")


if __name__ == "__main__":
    main()