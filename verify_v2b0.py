#!/usr/bin/env python3
"""
verify_student_v2b0_cv.py — Verify STUDENT (EfficientNetV2B0 + KD) qua nhiều SEED.

KHÁC verify_cv.py:
  - verify_cv.py    → verify TEACHER (EfficientNet-B0)
  - verify_student_v2b0_cv.py → verify STUDENT (EfficientNetV2B0 + KD distillation)

Mục đích: số student KD ổn định (mean ± std) cho thesis, không phải 1-run may rủi.

Pipeline GIỐNG train_student_kd.py:
  - Student: EfficientNetV2B0
  - KD: T=4.0, α=0.5 (từ config)
  - Teacher đã train sẵn (load từ TEACHER_PATH)
  - class weights xử lý imbalance

Chạy: source setup_gpu.sh && python3 verify_student_cv.py
(dùng DATASET_DIR trong config — đảm bảo = dataset_v4)
"""
import numpy as np
import subprocess
import sys
import json
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


# ── Student EfficientNetV2B0 (GIỐNG train_student_v2b0.py) ──
def build_student():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = True
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(C.NUM_CLASSES)(x)
    return models.Model(inputs, outputs, name="student_efficientnetv2b0")


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


def run_cv_student(paths, labels, teacher_path, seed):
    """1 lần CV (5-fold) cho student KD với 1 seed.
    Teacher load LẠI mỗi fold (vì clear_session xóa session cũ)."""
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

        # Load teacher MỚI cho fold này (session sạch sau clear_session)
        teacher = tf.keras.models.load_model(teacher_path)
        teacher.trainable = False

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


def _result_path(seed):
    """Đường dẫn file JSON tạm để tiến trình con báo kết quả về tiến trình cha."""
    return C.OUTPUT_DIR / f".verify_v2b0_seed_{seed}.json"


def run_single_seed_cli(seed):
    """Chạy CV cho ĐÚNG 1 seed rồi thoát. Dùng làm tiến trình CON (subprocess)
    → khi tiến trình này kết thúc, hệ điều hành thu hồi lại 100% RAM nó dùng,
    không phụ thuộc TensorFlow có dọn sạch bộ nhớ nội bộ hay không.
    Đây là cách chống OOM (Out-Of-Memory) triệt để nhất khi phải build
    nhiều model lớn liên tiếp trong cùng 1 lần chạy."""
    print(f"--- [subprocess] seed {seed} ---")
    if not C.TEACHER_PATH.exists():
        print(f"❌ KHÔNG tìm thấy teacher: {C.TEACHER_PATH}")
        sys.exit(1)

    paths, labels = gather_all()
    am, ast_, fm, fst = run_cv_student(paths, labels, C.TEACHER_PATH, seed)
    print(f"  Student KD acc: {am:.4f} ± {ast_:.4f} | F1: {fm:.4f} ± {fst:.4f}")

    with open(_result_path(seed), "w") as f:
        json.dump({"seed": seed, "acc_mean": am, "acc_std": ast_,
                    "f1_mean": fm, "f1_std": fst}, f)


def main():
    print("=" * 55)
    print("VERIFY STUDENT (EfficientNetV2B0 + KD)")
    print("=" * 55)
    print(f"Dataset: {C.DATASET_DIR}")
    print(f"KD: T={C.KD_TEMPERATURE}, α={C.KD_ALPHA}")
    print(f"Seeds: {SEEDS}")
    print("Mỗi seed chạy trong 1 tiến trình con riêng (chống OOM)\n")

    # Kiểm tra teacher tồn tại trước (báo lỗi sớm, khỏi tốn công spawn subprocess)
    print(f"📂 Teacher: {C.TEACHER_PATH}")
    if not C.TEACHER_PATH.exists():
        print(f"❌ KHÔNG tìm thấy teacher: {C.TEACHER_PATH}")
        print("   → train teacher trước (train_teacher.py)")
        return
    print("✓ Teacher OK\n")

    all_acc, all_f1 = [], []
    failed_seeds = []

    for seed in SEEDS:
        print(f"=== SEED {seed} (subprocess mới) ===")
        result_file = _result_path(seed)
        if result_file.exists():
            result_file.unlink()  # xoá kết quả cũ nếu có, tránh đọc nhầm

        # Chạy seed này trong 1 process con hoàn toàn mới.
        # KHÔNG capture_output → log của process con in thẳng ra terminal
        # (bạn vẫn thấy tiến trình train real-time như bình thường).
        proc = subprocess.run(
            [sys.executable, __file__, "--single-seed", str(seed)]
        )

        if proc.returncode != 0 or not result_file.exists():
            print(f"⚠️  Seed {seed} THẤT BẠI (có thể bị OOM kill, "
                  f"mã thoát={proc.returncode}) → bỏ qua, chạy tiếp seed sau")
            failed_seeds.append(seed)
            continue

        with open(result_file) as f:
            r = json.load(f)
        result_file.unlink()  # dọn file tạm

        all_acc.append(r["acc_mean"])
        all_f1.append(r["f1_mean"])

    if failed_seeds:
        print(f"\n⚠️  Các seed bị lỗi: {failed_seeds} "
              f"(nếu do OOM, thử giảm BATCH_SIZE trong config.py)")

    if not all_acc:
        print("\n❌ KHÔNG seed nào chạy thành công.")
        return

    print(f"\n{'='*55}")
    print(f"TỔNG HỢP {len(all_acc)}/{len(SEEDS)} seed thành công — STUDENT KD:")
    print(f"  Accuracy trung bình: {np.mean(all_acc):.4f}")
    print(f"  Std giữa các seed:   {np.std(all_acc):.4f}")
    print(f"  F1-macro trung bình: {np.mean(all_f1):.4f}")
    print(f"{'='*55}")
    if len(all_acc) < len(SEEDS):
        print("→ Một số seed bị bỏ qua — số liệu trên chỉ tính từ seed chạy được")
    elif np.std(all_acc) < 0.02:
        print("→ Ổn định qua các seed → con số student đáng tin cho thesis")
    else:
        print("→ Dao động giữa seed → ghi rõ variance trong thesis")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--single-seed":
        run_single_seed_cli(int(sys.argv[2]))
    else:
        main()