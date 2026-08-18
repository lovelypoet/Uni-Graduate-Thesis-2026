#!/usr/bin/env python3
"""
cross_validation.py — Stratified K-Fold CV cho teacher (data nhỏ).

Vì test set cố định quá nhỏ (~6 ảnh/class) → con số nhiễu.
CV gộp TOÀN BỘ data, chia K fold, train K lần, lấy trung bình ± std
→ con số ĐÁNG TIN để báo cáo trong thesis.

Chạy: python3 cross_validation.py [none|light|heavy]
      (mặc định: none nếu không truyền tham số)
"""
import sys
import numpy as np
import tensorflow as tf

# Cho phép TF cấp phát VRAM tăng dần (tránh OOM trên GPU 8GB)
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import config as C

N_FOLDS = 5
CV_EPOCHS_HEAD = 20      # giảm chút cho nhanh (CV train nhiều lần)
CV_EPOCHS_FT   = 15


# ──────────────────────────────────────────────────────────
# 1. Gom toàn bộ (path, label) từ train+val+test
# ──────────────────────────────────────────────────────────
def gather_all():
    paths, labels = [], []
    for split in ["train", "val", "test"]:
        for idx, cls in enumerate(C.CLASS_NAMES):
            d = C.DATASET_DIR / split / cls
            if not d.exists():
                continue
            for f in d.glob("*.jpg"):
                paths.append(str(f))
                labels.append(idx)
    return np.array(paths), np.array(labels)


# ──────────────────────────────────────────────────────────
# 2. tf.data từ danh sách path
# ──────────────────────────────────────────────────────────
def _decode(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
    return img, label


# Chế độ augmentation chọn qua tham số dòng lệnh:
#   python3 cross_validation.py none    → không augment (mặc định cũ)
#   python3 cross_validation.py light   → brightness/contrast nhẹ (đã có)
#   python3 cross_validation.py heavy   → + flip/rotate/zoom (MỚI, để lấy
#                                           lại số F1-macro còn thiếu trong
#                                           bảng ablation augmentation)
AUG_MODE = sys.argv[1] if len(sys.argv) > 1 else "none"


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(512, seed=C.SEED)
        if AUG_MODE == "light":
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomBrightness(0.10),
                tf.keras.layers.RandomContrast(0.10),
            ])
            ds = ds.map(lambda i, l: (aug(i, training=True), l),
                        num_parallel_calls=tf.data.AUTOTUNE)
        elif AUG_MODE == "heavy":
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.125),  # ±45 độ
                tf.keras.layers.RandomZoom(0.15),
                tf.keras.layers.RandomBrightness(0.10),
                tf.keras.layers.RandomContrast(0.10),
            ])
            ds = ds.map(lambda i, l: (aug(i, training=True), l),
                        num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────────────────
# 3. Model teacher (giống train_teacher.py)
# ──────────────────────────────────────────────────────────
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
    out = layers.Dense(C.NUM_CLASSES)(x)
    return models.Model(inp, out), bb


def class_weights_from(labels):
    total = len(labels)
    weights = {}
    for i in range(C.NUM_CLASSES):
        n = np.sum(labels == i)
        weights[i] = total / (C.NUM_CLASSES * n) if n > 0 else 1.0
    return weights


# ──────────────────────────────────────────────────────────
# 4. CV loop
# ──────────────────────────────────────────────────────────
def main():
    print(f"=== Augmentation mode: {AUG_MODE.upper()} ===\n")
    paths, labels = gather_all()
    print(f"Tổng số ảnh: {len(paths)}")
    for i, cls in enumerate(C.CLASS_NAMES):
        print(f"  {cls:11s}: {np.sum(labels==i)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=C.SEED)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    accs, f1s = [], []
    for fold, (trainval_idx, te_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\n{'='*45}\nFOLD {fold}/{N_FOLDS}\n{'='*45}")

        # Tách validation RIÊNG từ train fold (KHÔNG đụng test fold)
        # → test fold CHỈ dùng đánh giá cuối, tránh data leakage
        tv_paths = paths[trainval_idx]
        tv_labels = labels[trainval_idx]
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
        tr_sub, val_sub = next(inner.split(tv_paths, tv_labels))
        tr_idx = trainval_idx[tr_sub]
        val_idx = trainval_idx[val_sub]

        tr_ds  = make_ds(paths[tr_idx],  labels[tr_idx],  training=True)
        val_ds = make_ds(paths[val_idx], labels[val_idx], training=False)
        te_ds  = make_ds(paths[te_idx],  labels[te_idx],  training=False)
        cw = class_weights_from(labels[tr_idx])

        model, bb = build()
        # giai đoạn 1 — EarlyStopping dùng val_ds (KHÔNG phải te_ds)
        model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                      loss=loss_fn, metrics=["accuracy"])
        es = tf.keras.callbacks.EarlyStopping(
            patience=6, restore_best_weights=True,
            monitor="val_accuracy", mode="max")
        model.fit(tr_ds, validation_data=val_ds, epochs=CV_EPOCHS_HEAD,
                  class_weight=cw, callbacks=[es], verbose=0)
        # giai đoạn 2: fine-tune, BN frozen
        bb.trainable = True
        for l in bb.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(C.FINE_TUNE_LR),
                      loss=loss_fn, metrics=["accuracy"])
        model.fit(tr_ds, validation_data=val_ds, epochs=CV_EPOCHS_FT,
                  class_weight=cw, callbacks=[es], verbose=0)

        # đánh giá fold trên TEST FOLD (model chưa từng thấy)
        y_true, y_pred = [], []
        for x, y in te_ds:
            p = model(x, training=False)
            y_pred.extend(tf.argmax(p, axis=1).numpy())
            y_true.extend(y.numpy())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        accs.append(acc); f1s.append(f1)
        print(f"Fold {fold}: accuracy={acc:.4f}  F1-macro={f1:.4f}")

        # dọn VRAM kỹ giữa các fold (tránh tích lũy → OOM)
        del model, bb, tr_ds, val_ds, te_ds
        tf.keras.backend.clear_session()
        import gc; gc.collect()

    print(f"\n{'='*45}\nKẾT QUẢ {N_FOLDS}-FOLD CV\n{'='*45}")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print("\n→ Báo cáo con số TRUNG BÌNH ± STD này trong thesis")


if __name__ == "__main__":
    main()