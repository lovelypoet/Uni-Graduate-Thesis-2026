#!/usr/bin/env python3
"""
train_student_kd.py — Knowledge Distillation (Hinton et al., 2015).

Ý tưởng KD:
  Student học từ 2 nguồn:
    1. Hard label (nhãn thật)         → cross-entropy thường
    2. Soft label (output của teacher) → "dark knowledge"
  Loss = α · soft_loss + (1-α) · hard_loss

  soft_loss dùng nhiệt độ T để làm mềm phân phối:
    softmax(logits / T)  → teacher truyền "độ tự tin tương đối giữa các class"
    (vd: 'shift' dễ nhầm 'rotate' hơn 'missing' → student học được điều này)

Tạo CẢ HAI student để ablation:
  - student CÓ KD     → student_mobilenetv3_kd.keras
  - student KHÔNG KD  → student_mobilenetv3_nokd.keras (baseline)

Chạy: python3 train_student_kd.py
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

# Set seed để kết quả lặp lại được → so sánh aug công bằng
# (2 lần train chỉ khác nhau ở augmentation, không khác do random)
import numpy as np
tf.random.set_seed(C.SEED)
np.random.seed(C.SEED)


# ──────────────────────────────────────────────────────────
# Student: MobileNetV3-Large (cân bằng nhẹ + đủ capacity cho mobile)
# ──────────────────────────────────────────────────────────
def build_student():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV3Large(
        include_top=False, weights="imagenet", input_tensor=x
    )
    backbone.trainable = True   # student train hết từ đầu (đã có pretrained)
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(C.NUM_CLASSES)(x)
    return models.Model(inputs, outputs, name="student_mobilenetv3_large")


# ──────────────────────────────────────────────────────────
# Distiller: custom training loop gói trong Keras Model
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
            hard = self.hard_loss_fn(y, student_logits,
                                     sample_weight=sample_weight)
            soft_teacher = tf.nn.softmax(teacher_logits / self.T, axis=1)
            soft_student = tf.nn.softmax(student_logits / self.T, axis=1)
            soft = self.kl(soft_teacher, soft_student) * (self.T ** 2)
            loss = self.alpha * soft + (1 - self.alpha) * hard

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.student.trainable_variables))

        self.acc_metric.update_state(y, student_logits)
        self.loss_tracker.update_state(loss)
        return {"accuracy": self.acc_metric.result(),
                "loss": self.loss_tracker.result(),
                "hard": hard, "soft": soft}

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
def train_student_nokd(train, val, test, class_weights):
    """Baseline: student train thường, KHÔNG có teacher."""
    print("\n=== STUDENT KHÔNG KD (baseline) ===")
    student = build_student()
    student.compile(
        optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    cb = tf.keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True, monitor="val_accuracy")
    student.fit(train, validation_data=val, epochs=C.STUDENT_EPOCHS,
                class_weight=class_weights, callbacks=[cb])
    results = student.evaluate(test, return_dict=True)
    acc = results["accuracy"]
    print(f"🎯 Student (no KD) test accuracy: {acc:.4f}")
    student.save(C.STUDENT_NOKD_PATH)
    return acc


def train_student_kd(train, val, test, class_weights):
    """Student học qua Knowledge Distillation từ teacher."""
    print("\n=== STUDENT CÓ KD ===")
    teacher = tf.keras.models.load_model(C.TEACHER_PATH)
    teacher.trainable = False

    student = build_student()
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
    results = distiller.evaluate(test, return_dict=True)
    acc = results["accuracy"]
    print(f"🎯 Student (KD) test accuracy: {acc:.4f}")
    student.save(C.STUDENT_KD_PATH)
    return acc


def main():
    train, val, test = dl.get_datasets()
    class_weights, _ = dl.compute_class_weights()

    acc_nokd = train_student_nokd(train, val, test, class_weights)
    acc_kd   = train_student_kd(train, val, test, class_weights)

    print("\n" + "=" * 40)
    print(f"  Student KHÔNG KD: {acc_nokd:.4f}")
    print(f"  Student CÓ KD:    {acc_kd:.4f}")
    print(f"  KD cải thiện:     {(acc_kd - acc_nokd)*100:+.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()