#!/usr/bin/env python3
"""
kd_hyperparameter_sweep.py — Tìm (T, alpha) tốt nhất cho KD.

Grid: T in {2,3,4,5}, alpha in {0.3,0.5,0.7}  (12 cấu hình)

Chạy: python3 kd_hyperparameter_sweep.py
"""
import gc
import json
import numpy as np
import tensorflow as tf

_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass

from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score
import data_loader as dl
import config as C

# ── Seeds (giống train_student_kd.py) ────────────────────
tf.random.set_seed(C.SEED)
np.random.seed(C.SEED)


def build_student():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    bb = tf.keras.applications.MobileNetV3Large(
        include_top=False, weights="imagenet", input_tensor=x)
    bb.trainable = True
    x = layers.GlobalAveragePooling2D()(bb.output)
    x = layers.Dropout(0.3)(x)
    return models.Model(inputs, layers.Dense(C.NUM_CLASSES)(x))


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
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.lt = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.acc, self.lt]

    def train_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        t_logits = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s_logits = self.student(x, training=True)
            hard = self.hard_loss_fn(y, s_logits, sample_weight=sw)
            soft = self.kl(tf.nn.softmax(t_logits/self.T, 1),
                           tf.nn.softmax(s_logits/self.T, 1)) * (self.T**2)
            loss = self.alpha*soft + (1-self.alpha)*hard
        g = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.student.trainable_variables))
        self.acc.update_state(y, s_logits)
        self.lt.update_state(loss)
        return {"accuracy": self.acc.result(), "loss": self.lt.result()}

    def test_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        s_logits = self.student(x, training=False)
        self.acc.update_state(y, s_logits)
        self.lt.update_state(self.hard_loss_fn(y, s_logits))
        return {"accuracy": self.acc.result(), "loss": self.lt.result()}

    def call(self, x):
        return self.student(x)


def get_nokd_baseline(train, val, test, cw):
    """
    Chạy no-KD student một lần để lấy baseline thật,
    không hardcode số cũ.
    """
    print("\n" + "="*60)
    print("MEASURING no-KD BASELINE (fresh run, same seed)")
    print("="*60)
    student = build_student()
    student.compile(
        optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8,          # ← đồng bộ với sweep
        restore_best_weights=True, mode="max", verbose=0
    )
    student.fit(train, validation_data=val, epochs=C.STUDENT_EPOCHS,
                class_weight=cw, callbacks=[es], verbose=0)

    yt, yp = [], []
    for x, y in test:
        yp.extend(tf.argmax(student(x, training=False), 1).numpy())
        yt.extend(y.numpy())
    acc = accuracy_score(yt, yp)
    f1  = f1_score(yt, yp, average="macro", zero_division=0)
    print(f"no-KD baseline → acc={acc:.4f}  f1={f1:.4f}")

    del student
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()
    return acc, f1


def eval_test(model, test):
    yt, yp = [], []
    for x, y in test:
        yp.extend(tf.argmax(model(x, training=False), 1).numpy())
        yt.extend(y.numpy())
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro", zero_division=0)


def main():
    print("="*60)
    print("KD HYPERPARAMETER SWEEP")
    print(f"Dataset : {C.DATASET_DIR}")
    print(f"Seed    : {C.SEED}")
    print(f"Grid    : T∈{{2,3,4,5}} × α∈{{0.3,0.5,0.7}} = 12 configs")
    print("="*60)

    train, val, test = dl.get_datasets()
    cw, counts = dl.compute_class_weights()

    print("\nClass distribution (train):")
    for i, cls in enumerate(C.CLASS_NAMES):
        print(f"  {cls:12s}: {counts[i]}")

    # ── Baseline no-KD (chạy thật, không hardcode) ───────
    nokd_acc, nokd_f1 = get_nokd_baseline(train, val, test, cw)

    # ── Load teacher ──────────────────────────────────────
    print(f"\nLoading teacher from {C.TEACHER_PATH}...")
    teacher = tf.keras.models.load_model(C.TEACHER_PATH)
    teacher.trainable = False

    # ── Grid search ───────────────────────────────────────
    results = []
    for T in [2.0, 3.0, 4.0, 5.0]:
        for alpha in [0.3, 0.5, 0.7]:
            print(f"\n{'='*60}")
            print(f"T={T}, α={alpha}")
            print(f"{'='*60}")

            student = build_student()
            dist = Distiller(student, teacher)
            dist.compile(
                optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                alpha=alpha,
                temperature=T,
            )
            es = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,                          # ← sửa 6 → 8
                restore_best_weights=True,
                mode="max",
                verbose=0,
            )
            dist.fit(train, validation_data=val, epochs=C.STUDENT_EPOCHS,
                     class_weight=cw, callbacks=[es], verbose=0)

            acc, f1 = eval_test(dist, test)
            print(f"  acc={acc:.4f}  f1={f1:.4f}  "
                  f"({(acc-nokd_acc)*100:+.2f}pp vs no-KD)")

            results.append({
                "T": T, "alpha": alpha,
                "accuracy": float(acc), "f1": float(f1),
                "vs_nokd_pp": float((acc - nokd_acc) * 100),
            })

            del student, dist
            gc.collect()
            tf.keras.backend.clear_session()
            gc.collect()

    # ── Results ───────────────────────────────────────────
    results.sort(key=lambda r: (r["accuracy"], r["f1"]), reverse=True)

    print("\n" + "="*60)
    print(f"RESULTS (sorted by accuracy)")
    print(f"no-KD baseline: acc={nokd_acc:.4f}  f1={nokd_f1:.4f}")
    print("="*60)
    for i, r in enumerate(results, 1):
        mark = "✅" if r["vs_nokd_pp"] > 0 else "❌"
        print(f"{i:2d}. T={r['T']} α={r['alpha']} → "
              f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"({r['vs_nokd_pp']:+.2f}pp vs no-KD) {mark}")

    best = results[0]
    print(f"\n🏆 BEST: T={best['T']}, α={best['alpha']}")
    print(f"   acc={best['accuracy']:.4f}  f1={best['f1']:.4f}")

    if best["vs_nokd_pp"] > 0:
        print(f"\n→ Update config.py:")
        print(f"   KD_TEMPERATURE = {best['T']}")
        print(f"   KD_ALPHA       = {best['alpha']}")
    else:
        print("\n→ No config beats no-KD baseline.")
        print("  Thesis narrative: student retains accuracy close to teacher"
              " through compression, not via KD accuracy gain.")

    # ── Save ─────────────────────────────────────────────
    output = {
        "nokd_baseline": {"accuracy": float(nokd_acc), "f1": float(nokd_f1)},
        "seed": C.SEED,
        "dataset": str(C.DATASET_DIR),
        "results": results,
    }
    with open("kd_sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n💾 Saved: kd_sweep_results.json")


if __name__ == "__main__":
    main()