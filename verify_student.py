#!/usr/bin/env python3
"""
verify_student_cv.py — Verify STUDENT (MobileNetV3-Large + KD) qua nhiều SEED.

KHÁC verify_cv.py:
  - verify_cv.py    → verify TEACHER (EfficientNet-B0)
  - verify_student_cv.py → verify STUDENT (MobileNetV3 + KD distillation)

Mục đích: số student KD ổn định (mean ± std) cho thesis, không phải 1-run may rủi.

Pipeline GIỐNG train_student_kd.py:
  - Student: MobileNetV3-Large
  - KD: T=4.0, α=0.5 (từ config)
  - Teacher đã train sẵn (load từ TEACHER_PATH)
  - class weights xử lý imbalance

Chạy: source setup_gpu.sh && python3 verify_student_cv.py
(dùng DATASET_DIR trong config — đảm bảo = dataset_v4)
"""
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import gc
import config as C

N_FOLDS = 5
STUDENT_EPOCHS_CV = 30      # ít hơn full train (40) để verify nhanh hơn
SEEDS = [42, 123, 2024]     # 3 seed giống verify teacher


# ──────────────────────────────────────────────────────────
def gather_all():
    """Gom tất cả ảnh (train+val+test) để CV chia lại."""
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


# ── Student MobileNetV3-Large (GIỐNG train_student_kd.py) ──
def build_student():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV3Large(
        include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = True
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(C.NUM_CLASSES)(x)
    return models.Model(inputs, outputs, name="student_mobilenetv3")


# ── Distiller (GIỐNG train_student_kd.py) ──
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


def cw_from(labels):
    total = len(labels)
    return {i: (total/(C.NUM_CLASSES*np.sum(labels==i))
                if np.sum(labels==i)>0 else 1.0) for i in range(C.NUM_CLASSES)}


def run_cv_student(paths, labels, teacher, seed):
    """1 lần CV (5-fold) cho student KD với 1 seed."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for trainval_idx, te_idx in skf.split(paths, labels):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        tr_sub, val_sub = next(inner.split(paths[trainval_idx], labels[trainval_idx]))
        tr_idx, val_idx = trainval_idx[tr_sub], trainval_idx[val_sub]
        tr = make_ds(paths[tr_idx], labels[tr_idx], True, seed)
        va = make_ds(paths[val_idx], labels[val_idx], False, seed)
        te = make_ds(paths[te_idx], labels[te_idx], False, seed)
        cw = cw_from(labels[tr_idx])

        student = build_student()
        distiller = Distiller(student, teacher)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
            alpha=C.KD_ALPHA, temperature=C.KD_TEMPERATURE)
        es = tf.keras.callbacks.EarlyStopping(
            patience=6, restore_best_weights=True,
            monitor="val_accuracy", mode="max")
        distiller.fit(tr, validation_data=va, epochs=STUDENT_EPOCHS_CV,
                      class_weight=cw, callbacks=[es], verbose=0)

        # đánh giá student trên test fold
        yt, yp = [], []
        for x, y in te:
            yp.extend(tf.argmax(student(x, training=False), 1).numpy())
            yt.extend(y.numpy())
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))

        tf.keras.backend.clear_session()
        gc.collect()
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def main():
    print("=" * 55)
    print("VERIFY STUDENT (MobileNetV3-Large + KD)")
    print("=" * 55)
    print(f"Dataset: {C.DATASET_DIR}")
    print(f"KD: T={C.KD_TEMPERATURE}, α={C.KD_ALPHA}")
    print(f"Seeds: {SEEDS}\n")

    # load teacher 1 lần (dùng chung cho mọi fold/seed)
    print(f"📂 Loading teacher: {C.TEACHER_PATH}")
    teacher = tf.keras.models.load_model(C.TEACHER_PATH)
    teacher.trainable = False
    print("✓ Teacher loaded\n")

    paths, labels = gather_all()
    print(f"Tổng ảnh: {len(paths)}\n")

    all_acc, all_f1 = [], []
    for seed in SEEDS:
        print(f"=== SEED {seed} ===")
        am, ast_, fm, fst = run_cv_student(paths, labels, teacher, seed)
        print(f"  Student KD acc: {am:.4f} ± {ast_:.4f} | F1: {fm:.4f} ± {fst:.4f}")
        all_acc.append(am)
        all_f1.append(fm)

    print(f"\n{'='*55}")
    print(f"TỔNG HỢP {len(SEEDS)} seed — STUDENT KD:")
    print(f"  Accuracy trung bình: {np.mean(all_acc):.4f}")
    print(f"  Std giữa các seed:   {np.std(all_acc):.4f}")
    print(f"  F1-macro trung bình: {np.mean(all_f1):.4f}")
    print(f"{'='*55}")
    if np.std(all_acc) < 0.02:
        print("→ Ổn định qua các seed → con số student đáng tin cho thesis")
    else:
        print("→ Dao động giữa seed → ghi rõ variance trong thesis")


if __name__ == "__main__":
    main()