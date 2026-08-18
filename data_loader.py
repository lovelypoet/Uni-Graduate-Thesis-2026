#!/usr/bin/env python3
"""
data_loader.py — Load dataset + augmentation + class weights.

Augmentation strategy (dựa trên thực nghiệm):
- TOÀN BỘ class: brightness/contrast nhẹ (safe, không đổi nhãn)
- CLASS KHÔNG NHẠY GÓC (ok, missing, bridge):
    + Thêm flip + rotation (an toàn vì nhãn không phụ thuộc góc)
- CLASS NHẠY GÓC (rotate, shift, tilt, tombstone):
    + KHÔNG xoay/lật (vì xoay "ok" → trông như "rotate", v.v.)

Tắt/bật augmentation qua config:
    USE_AUGMENTATION = True/False
    USE_ANGLE_AUG = True/False   ← mới, bật/tắt phần góc độ
"""
import tensorflow as tf
import numpy as np
from collections import Counter
import config as C


# ──────────────────────────────────────────────────────────
# 1. Load split từ folder
# ──────────────────────────────────────────────────────────
def _load_split(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        C.DATASET_DIR / split,
        labels="inferred",
        label_mode="int",
        class_names=C.CLASS_NAMES,
        image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=C.BATCH_SIZE,
        shuffle=shuffle,
        seed=C.SEED,
    )


# ──────────────────────────────────────────────────────────
# 2. Augmentation layers
# ──────────────────────────────────────────────────────────

# Safe aug: áp cho TẤT CẢ class (không đổi nhãn)
_safe_aug = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness(0.10),
    tf.keras.layers.RandomContrast(0.10),
], name="safe_aug")

# Angle aug: CHỈ cho class không nhạy góc (ok, missing, bridge)
_angle_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.25),   # ±45° (0.25 × 360° = 90°)
], name="angle_aug")

# Index của class NHẠY GÓC → sẽ KHÔNG áp angle aug
_ANGLE_SENSITIVE_IDX = tf.constant(
    [C.CLASS_NAMES.index(c) for c in C.ANGLE_SENSITIVE],
    dtype=tf.int32,
)


def _augment(images, labels):
    """
    Augment một batch:
    1. Safe aug (brightness/contrast) → áp toàn bộ
    2. Angle aug (flip/rotate) → chỉ ảnh có class KHÔNG nhạy góc
    """
    # Bước 1: safe aug cho tất cả
    images = _safe_aug(images, training=True)

    # Bước 2: angle aug có điều kiện
    if getattr(C, "USE_ANGLE_AUG", False):
        labels_i32 = tf.cast(labels, tf.int32)

        # is_sensitive[i] = True nếu ảnh thứ i thuộc class nhạy góc
        is_sensitive = tf.reduce_any(
            tf.equal(
                tf.expand_dims(labels_i32, 1),   # (B, 1)
                _ANGLE_SENSITIVE_IDX,             # (n_sensitive,)
            ),
            axis=1,
        )  # shape (B,)

        # Tạo bản đã xoay/lật cho toàn batch
        augmented = _angle_aug(images, training=True)

        # Chọn: nhạy góc → giữ nguyên, không nhạy → dùng bản augmented
        mask = tf.reshape(is_sensitive, (-1, 1, 1, 1))
        images = tf.where(mask, images, augmented)

    return images, labels


# ──────────────────────────────────────────────────────────
# 3. Class weights
# ──────────────────────────────────────────────────────────
def compute_class_weights():
    counts = Counter()
    train_dir = C.DATASET_DIR / "train"
    for i, cls in enumerate(C.CLASS_NAMES):
        n = len(list((train_dir / cls).glob("*.jpg")))
        counts[i] = n
    total = sum(counts.values())
    weights = {i: total / (C.NUM_CLASSES * counts[i]) for i in counts}
    return weights, counts


# ──────────────────────────────────────────────────────────
# 4. API chính
# ──────────────────────────────────────────────────────────
def get_datasets():
    AUTOTUNE = tf.data.AUTOTUNE

    train = _load_split("train", shuffle=True)
    val   = _load_split("val",   shuffle=False)
    test  = _load_split("test",  shuffle=False)

    use_aug   = getattr(C, "USE_AUGMENTATION", True)
    use_angle = getattr(C, "USE_ANGLE_AUG", False)

    if use_aug:
        train = train.map(_augment, num_parallel_calls=AUTOTUNE)
        if use_angle:
            print("[data_loader] Augmentation: brightness/contrast + flip/rotate "
                  "(angle aug CHỈ cho ok/missing/bridge)")
        else:
            print("[data_loader] Augmentation: brightness/contrast nhẹ (safe only)")
    else:
        print("[data_loader] Augmentation: TẮT")

    train = train.prefetch(AUTOTUNE)
    val   = val.prefetch(AUTOTUNE)
    test  = test.prefetch(AUTOTUNE)
    return train, val, test


# ──────────────────────────────────────────────────────────
# Test nhanh
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Class:", C.CLASS_NAMES)
    print("Angle-sensitive:", C.ANGLE_SENSITIVE)
    print("USE_ANGLE_AUG:", getattr(C, "USE_ANGLE_AUG", False))

    weights, counts = compute_class_weights()
    print("\nSố ảnh train mỗi class:")
    for i, cls in enumerate(C.CLASS_NAMES):
        print(f"  {cls:12s}: {counts[i]:4d}  (weight {weights[i]:.2f})")

    train, val, test = get_datasets()
    xb, yb = next(iter(train))
    print(f"\nBatch shape: {xb.shape}, dtype {xb.dtype}")
    print(f"Label range: {yb.numpy().min()}–{yb.numpy().max()}")
    print("✅ data_loader OK")