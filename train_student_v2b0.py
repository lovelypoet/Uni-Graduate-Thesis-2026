#!/usr/bin/env python3
"""
train_student_v2b0.py — Thử student EfficientNetV2B0 (thay MobileNetV3-Large).

Mục đích: so sánh student backbone khác cho thesis.
  - MobileNetV3-Large (gốc): 5.4M params, feature 7x7x960
  - EfficientNetV2B0 (mới):   5.9M params, feature 7x7x1280
  → V2B0 thiết kế tối ưu cho edge, feature map nhiều channel hơn
    → có thể Grad-CAM mịn hơn + accuracy tốt hơn

KD pipeline GIỮ NGUYÊN (T=4, α=0.5), chỉ đổi backbone student.

Chạy: python3 train_student_v2b0.py
Output: student_v2b0_kd.keras + student_v2b0_nokd.keras
"""
import tensorflow as tf

_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass

from tensorflow.keras import layers, models
import numpy as np
import gc
import data_loader as dl
import config as C

tf.random.set_seed(C.SEED)
np.random.seed(C.SEED)


# ──────────────────────────────────────────────────────────
# Student: EfficientNetV2B0 (thay MobileNetV3-Large)
# ──────────────────────────────────────────────────────────
def build_student_v2b0():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    # V2 có preprocess riêng (KHÁC MobileNetV3)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_tensor=x
    )
    backbone.trainable = True
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(C.NUM_CLASSES)(x)
    return models.Model(inputs, outputs, name="student_efficientnetv2b0")


# ──────────────────────────────────────────────────────────
# Distiller (GIỐNG train_student_kd.py — KD pipeline không đổi)
# ──────────────────────────────────────────────────────────
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, alpha, temperature):
        super().compile(optimizer=optimizer)
        self.alpha = alpha
        self.T = temperature
        self.hard_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.kl = tf.keras.losses.KLDivergence()
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.acc_metric, self.loss_tracker]

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        teacher_logits = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            hard = self.hard_loss_fn(y, student_logits, sample_weight=sample_weight)
            soft_teacher = tf.nn.softmax(teacher_logits / self.T, axis=1)
            soft_student = tf.nn.softmax(student_logits / self.T, axis=1)
            soft = self.kl(soft_teacher, soft_student) * (self.T ** 2)
            loss = self.alpha * soft + (1 - self.alpha) * hard
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.acc_metric.update_state(y, student_logits)
        self.loss_tracker.update_state(loss)
        return {"accuracy": self.acc_metric.result(),
                "loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        student_logits = self.student(x, training=False)
        loss = self.hard_loss_fn(y, student_logits)
        self.acc_metric.update_state(y, student_logits)
        self.loss_tracker.update_state(loss)
        return {"accuracy": self.acc_metric.result(),
                "loss": self.loss_tracker.result()}

    def call(self, x):
        return self.student(x)


# ──────────────────────────────────────────────────────────
def train_nokd(train, val, test, class_weights):
    print("\n=== STUDENT V2B0 KHÔNG KD (baseline) ===")
    student = build_student_v2b0()
    student.compile(
        optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    cb = tf.keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True, monitor="val_accuracy")
    student.fit(train, validation_data=val, epochs=C.STUDENT_EPOCHS,
                class_weight=class_weights, callbacks=[cb])
    acc = student.evaluate(test, return_dict=True)["accuracy"]
    print(f"🎯 V2B0 (no KD) test accuracy: {acc:.4f}")
    student.save(C.OUTPUT_DIR / "student_v2b0_nokd.keras")
    return acc


def train_kd(train, val, test, class_weights):
    print("\n=== STUDENT V2B0 CÓ KD ===")
    teacher = tf.keras.models.load_model(C.TEACHER_PATH)
    teacher.trainable = False
    student = build_student_v2b0()
    distiller = Distiller(student, teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
        alpha=C.KD_ALPHA,
        temperature=C.KD_TEMPERATURE,
    )
    cb = tf.keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True,
        monitor="val_accuracy", mode="max")
    distiller.fit(train, validation_data=val, epochs=C.STUDENT_EPOCHS,
                  class_weight=class_weights, callbacks=[cb])
    acc = distiller.evaluate(test, return_dict=True)["accuracy"]
    print(f"🎯 V2B0 (KD) test accuracy: {acc:.4f}")
    student.save(C.OUTPUT_DIR / "student_v2b0_kd.keras")
    return acc


def main():
    train, val, test = dl.get_datasets()
    class_weights, _ = dl.compute_class_weights()

    acc_nokd = train_nokd(train, val, test, class_weights)

    # cleanup tránh OOM giữa 2 lần train
    tf.keras.backend.clear_session()
    gc.collect()

    acc_kd = train_kd(train, val, test, class_weights)

    print("\n" + "=" * 45)
    print("  STUDENT: EfficientNetV2B0 (5.9M params)")
    print(f"  KHÔNG KD: {acc_nokd:.4f}")
    print(f"  CÓ KD:    {acc_kd:.4f}")
    print(f"  KD cải thiện: {(acc_kd - acc_nokd)*100:+.2f}%")
    print("=" * 45)
    print("\n→ So sánh với MobileNetV3-Large (KD ~0.84) để chọn student tốt hơn")


if __name__ == "__main__":
    main()