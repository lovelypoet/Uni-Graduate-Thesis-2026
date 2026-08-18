#!/usr/bin/env python3
"""
cross_validation_v2.py — 5-Fold CV dùng EfficientNet-B2 (để so sánh với B0).

Dùng để lấy con số B2 chính thức cho bảng ablation "B0 vs B2".
Đã sửa data leakage: test fold CHỈ dùng đánh giá cuối, validation
tách riêng từ train fold (giống cross_validation.py).

Chạy: source setup_gpu.sh && python3 cross_validation_v2.py [none|light|heavy]
      (mặc định: light nếu không truyền tham số, để giữ hành vi cũ)
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

tf.random.set_seed(C.SEED)
np.random.seed(C.SEED)

N_FOLDS = 5
CV_EPOCHS_HEAD = 8
CV_EPOCHS_FT   = 25


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


# Chế độ augmentation chọn qua tham số dòng lệnh:
#   python3 cross_validation_v2.py none    → không augment
#   python3 cross_validation_v2.py light   → brightness/contrast nhẹ (mặc định cũ)
#   python3 cross_validation_v2.py heavy   → + flip/rotate/zoom (MỚI, để lấy
#                                              lại số F1-macro còn thiếu)
AUG_MODE = sys.argv[1] if len(sys.argv) > 1 else "light"


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


def build_teacher_b2():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    backbone = tf.keras.applications.EfficientNetB2(
        include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = False
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(C.NUM_CLASSES)(x)
    return models.Model(inputs, outputs, name="teacher_effnetb2"), backbone


def freeze_batchnorm(backbone):
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def class_weights_from(labels):
    total = len(labels)
    return {i: (total / (C.NUM_CLASSES * np.sum(labels == i))
                if np.sum(labels == i) > 0 else 1.0)
            for i in range(C.NUM_CLASSES)}


def main():
    print("5-Fold CV — EfficientNet-B2 (so sánh với B0)")
    print(f"=== Augmentation mode: {AUG_MODE.upper()} ===\n")
    paths, labels = gather_all()
    print(f"Tổng ảnh: {len(paths)}")
    for i, cls in enumerate(C.CLASS_NAMES):
        print(f"  {cls:11s}: {np.sum(labels==i)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=C.SEED)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    accs, f1s = [], []
    for fold, (trainval_idx, te_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"\n{'='*50}\nFOLD {fold}/{N_FOLDS}\n{'='*50}")

        # tách validation RIÊNG từ train fold (không đụng test fold)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
        tr_sub, val_sub = next(inner.split(paths[trainval_idx],
                                           labels[trainval_idx]))
        tr_idx = trainval_idx[tr_sub]
        val_idx = trainval_idx[val_sub]

        tr_ds  = make_ds(paths[tr_idx],  labels[tr_idx],  training=True)
        val_ds = make_ds(paths[val_idx], labels[val_idx], training=False)
        te_ds  = make_ds(paths[te_idx],  labels[te_idx],  training=False)
        cw = class_weights_from(labels[tr_idx])
        print(f"Train: {len(tr_idx)} | Val: {len(val_idx)} | Test: {len(te_idx)}")

        model, bb = build_teacher_b2()
        # giai đoạn 1
        model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                      loss=loss_fn, metrics=["accuracy"])
        es = tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True,
            monitor="val_accuracy", mode="max", verbose=0)
        model.fit(tr_ds, validation_data=val_ds, epochs=CV_EPOCHS_HEAD,
                  class_weight=cw, callbacks=[es], verbose=0)
        # giai đoạn 2: fine-tune BN frozen
        bb.trainable = True
        freeze_batchnorm(bb)
        model.compile(optimizer=tf.keras.optimizers.Adam(C.FINE_TUNE_LR),
                      loss=loss_fn, metrics=["accuracy"])
        model.fit(tr_ds, validation_data=val_ds, epochs=CV_EPOCHS_FT,
                  class_weight=cw, callbacks=[es], verbose=0)

        # đánh giá trên TEST FOLD (chưa thấy)
        y_true, y_pred = [], []
        for x, y in te_ds:
            p = model(x, training=False)
            y_pred.extend(tf.argmax(p, axis=1).numpy())
            y_true.extend(y.numpy())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accs.append(acc); f1s.append(f1)
        print(f"Fold {fold}: accuracy={acc:.4f}  F1-macro={f1:.4f}")
        del model, bb
        tf.keras.backend.clear_session()

    print(f"\n{'='*50}\nKẾT QUẢ 5-FOLD CV (B2)\n{'='*50}")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1-macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print("\nPer-fold:")
    for i, (a, f) in enumerate(zip(accs, f1s), 1):
        print(f"  Fold {i}: acc={a:.4f}  f1={f:.4f}")


if __name__ == "__main__":
    main()