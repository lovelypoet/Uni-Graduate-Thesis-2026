#!/usr/bin/env python3
"""
train_teacher.py — Train teacher EfficientNet-B0 (BẢN CẢI TIẾN cho data nhỏ).

Các fix quan trọng so với bản đầu:
  1. BatchNorm GIỮ FROZEN khi fine-tune (lỗi kinh điển của EfficientNet:
     fine-tune cả BN trên data nhỏ → phá thống kê BN → accuracy tụt).
  2. Head mạnh hơn (thêm Dense + BN) để học đặc trưng PCB.
  3. Fine-tune mở khoá backbone (PCB rất khác ImageNet nên cần fine-tune sâu).
  4. Nhiều epoch + patience cao hơn.

Chạy: python3 train_teacher.py
"""
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
import data_loader as dl
import config as C

# Set seed để kết quả lặp lại được (so sánh công bằng)
import numpy as np
tf.random.set_seed(C.SEED)
np.random.seed(C.SEED)


def build_teacher():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = False
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(C.NUM_CLASSES)(x)
    model = models.Model(inputs, outputs, name="teacher_effnetb0")
    return model, backbone


def freeze_batchnorm(backbone):
    """Giữ tất cả BatchNorm ở chế độ inference (không cập nhật thống kê)."""
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def main():
    train, val, test = dl.get_datasets()
    class_weights, _ = dl.compute_class_weights()

    model, backbone = build_teacher()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Giai đoạn 1: train head (backbone frozen)
    print("\n=== Giai đoạn 1: train head (backbone frozen) ===")
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                  loss=loss, metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                         monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5,
                                             monitor="val_loss"),
    ]
    model.fit(train, validation_data=val, epochs=C.TEACHER_EPOCHS,
              class_weight=class_weights, callbacks=cbs)

    # Giai đoạn 2: fine-tune (mở backbone, GIỮ BN frozen)
    print("\n=== Giai đoạn 2: fine-tune (BN giu frozen) ===")
    backbone.trainable = True
    freeze_batchnorm(backbone)          # fix quan trong nhat
    model.compile(optimizer=tf.keras.optimizers.Adam(C.FINE_TUNE_LR),
                  loss=loss, metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=25,
              class_weight=class_weights, callbacks=cbs)

    test_loss, test_acc = model.evaluate(test)
    print(f"\nTeacher test accuracy: {test_acc:.4f}")
    model.save(C.TEACHER_PATH)
    print(f"Da luu: {C.TEACHER_PATH}")
    print("\n-> Chay 'python3 evaluate.py outputs/teacher_effnetb0.keras' "
          "de xem confusion matrix")


if __name__ == "__main__":
    main()