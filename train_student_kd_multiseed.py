#!/usr/bin/env python3
"""
train_student_kd_multiseed.py — Multi-seed Knowledge Distillation.

Mục đích:
  Bản single-run cũ cho KD gain dao động rất mạnh giữa các lần chạy
  (+4.6% rồi -3.45% với cùng config). Đó là dấu hiệu variance cao do
  early stopping trên val set nhỏ + non-determinism GPU. Bản này chạy
  cùng cặp no-KD / KD trên NHIỀU seed cố định, rồi báo cáo mean ± std
  cho CẢ HAI — con số trung thực, tái lập được, để đưa vào thesis.

  QUAN TRỌNG: báo cáo TOÀN BỘ kết quả bất kể ra sao. Không chọn seed
  đẹp. Bộ seed {42, 123, 2024} cố định trước, giống seed đã dùng cho
  teacher-verify, để nhất quán toàn thesis.

Khác biệt so với train_student_kd.py:
  - Set seed LẠI đầu mỗi iteration (bản cũ set 1 lần lúc import → vô tác dụng)
  - KHÔNG lưu model (tránh ghi đè model đang deploy); đây là run đo đạc
  - In per-seed + mean ± std + gain từng seed

Chạy: python3 train_student_kd_multiseed.py
"""
import os
import random
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
import data_loader as dl
import config as C

# Bộ seed cố định — QUYẾT ĐỊNH TRƯỚC, báo cáo hết, không cherry-pick.
SEEDS = [42, 123, 2024]


def set_all_seeds(seed):
    """Set seed cho mọi nguồn ngẫu nhiên. Gọi TRƯỚC khi build model."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ──────────────────────────────────────────────────────────
# Student: MobileNetV3-Large
# ──────────────────────────────────────────────────────────
def build_student():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV3Large(
        include_top=False, weights="imagenet", input_tensor=x
    )
    backbone.trainable = True
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(C.NUM_CLASSES)(x)
    return models.Model(inputs, outputs, name="student_mobilenetv3_large")


# ──────────────────────────────────────────────────────────
# Distiller (giữ nguyên logic bản gốc)
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
# Train một lần (một seed) — KHÔNG lưu model
# ──────────────────────────────────────────────────────────
def run_nokd(train, val, test, class_weights):
    student = build_student()
    student.compile(
        optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    cb = tf.keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True, monitor="val_accuracy",
        mode="max")
    student.fit(train, validation_data=val, epochs=C.STUDENT_EPOCHS,
                class_weight=class_weights, callbacks=[cb], verbose=0)
    return student.evaluate(test, return_dict=True, verbose=0)["accuracy"]


def run_kd(train, val, test, class_weights, teacher):
    student = build_student()
    distiller = Distiller(student, teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
        alpha=C.KD_ALPHA,
        temperature=C.KD_TEMPERATURE,
    )
    cb = tf.keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True, monitor="val_accuracy",
        mode="max")
    distiller.fit(train, validation_data=val, epochs=C.STUDENT_EPOCHS,
                  class_weight=class_weights, callbacks=[cb], verbose=0)
    return distiller.evaluate(test, return_dict=True, verbose=0)["accuracy"]


# ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MULTI-SEED KD  |  seeds =", SEEDS)
    print(f"Dataset: {C.DATASET_DIR}  |  Batch: {C.BATCH_SIZE}  |  "
          f"T={C.KD_TEMPERATURE}, alpha={C.KD_ALPHA}")
    print("=" * 60)

    # Teacher load MỘT LẦN (nó frozen, không phụ thuộc seed student)
    teacher = tf.keras.models.load_model(C.TEACHER_PATH)
    teacher.trainable = False

    class_weights, _ = dl.compute_class_weights()

    nokd_accs, kd_accs, gains = [], [], []

    for seed in SEEDS:
        print(f"\n{'─'*60}\nSEED {seed}\n{'─'*60}")
        set_all_seeds(seed)

        # Reload dataset sau khi set seed (để shuffle/aug theo seed này).
        # Split train/val/test cố định trên đĩa → test luôn giống nhau.
        train, val, test = dl.get_datasets()

        acc_nokd = run_nokd(train, val, test, class_weights)
        acc_kd = run_kd(train, val, test, class_weights, teacher)
        gain = (acc_kd - acc_nokd) * 100

        nokd_accs.append(acc_nokd)
        kd_accs.append(acc_kd)
        gains.append(gain)
        print(f"  no-KD: {acc_nokd:.4f}  |  KD: {acc_kd:.4f}  |  "
              f"gain: {gain:+.2f}pp")

    # ── Tổng hợp — báo cáo TOÀN BỘ, không giấu seed nào ──
    nokd_accs = np.array(nokd_accs)
    kd_accs = np.array(kd_accs)
    gains = np.array(gains)

    print(f"\n{'='*60}\nKẾT QUẢ MULTI-SEED ({len(SEEDS)} seeds)\n{'='*60}")
    print("Per-seed:")
    for s, n, k, g in zip(SEEDS, nokd_accs, kd_accs, gains):
        print(f"  seed {s:>4}:  no-KD={n:.4f}  KD={k:.4f}  gain={g:+.2f}pp")
    print("-" * 60)
    print(f"no-KD : {nokd_accs.mean():.4f} ± {nokd_accs.std():.4f}")
    print(f"KD    : {kd_accs.mean():.4f} ± {kd_accs.std():.4f}")
    print(f"gain  : {gains.mean():+.2f}pp ± {gains.std():.2f}pp")
    print("=" * 60)
    print("\n→ Báo cáo mean ± std này trong thesis, BẤT KỂ nó ra sao.")
    print("  Nếu gain std nuốt trọn mean (vd +1.5 ± 4.0) → nói thẳng là")
    print("  KD gain trên v1 không ổn định về mặt thống kê, đúng như")
    print("  lập luận teacher–student gap nhỏ ở phần Discussion.")


if __name__ == "__main__":
    main()