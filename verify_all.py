#!/usr/bin/env python3
"""
verify_student_nokd_vs_kd_cv.py — So sánh no-KD vs KD bằng 5-fold CV, nhiều seed.

Dựa trên verify_student_cv.py của bạn (giữ nguyên CV protocol chống leakage:
test fold cô lập, val tách từ inner split). Khác biệt DUY NHẤT: mỗi fold train
CẢ HAI student trên CÙNG một split —
  - no-KD: student MobileNetV3-Large train thuần (hard label)
  - KD   : cùng student + distill từ teacher (T, α từ config)
→ gain = KD − no-KD đo trên đúng cùng dữ liệu, so sánh công bằng tuyệt đối.

Đây là run "chốt" để trả lời DỨT KHOÁT: trên dataset này, KD có hơn no-KD không?
Báo cáo TOÀN BỘ per-seed + mean ± std, BẤT KỂ kết quả ra sao (không cherry-pick).

Chạy cho v1:  đặt config → DATASET_DIR="dataset", TEACHER_PATH=teacher v1, BATCH=32
Chạy cho v4:  đặt config → DATASET_DIR="dataset_v4", TEACHER_PATH=teacher v4, BATCH=16
    python3 verify_student_nokd_vs_kd_cv.py 2>&1 | tee nokd_vs_kd_<ver>.log
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
STUDENT_EPOCHS_CV = 30
SEEDS = [42, 123, 2024]


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


def make_ds(paths, labels, training, seed):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_decode, num_parallel_calls=1)
    if training:
        ds = ds.shuffle(min(len(paths), 128), seed=seed, reshuffle_each_iteration=True)
        if getattr(C, "USE_AUGMENTATION", True):
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomBrightness(0.10),
                tf.keras.layers.RandomContrast(0.10)])
            ds = ds.map(lambda i, l: (aug(i, training=True), l),
                        num_parallel_calls=1)
    return ds.batch(C.BATCH_SIZE).prefetch(1)


def build_student():
    inputs = layers.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    # [Yêu cầu 8] TF sẽ tải weights "imagenet" lần đầu và tự động cache ở ~/.keras/models, hoàn toàn bình thường.
    backbone = tf.keras.applications.MobileNetV3Large(
        include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = True
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(C.NUM_CLASSES)(x)
    return models.Model(inputs, outputs, name="student_mobilenetv3")


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
                if np.sum(labels==i) > 0 else 1.0) for i in range(C.NUM_CLASSES)}


def _eval(student, te):
    yt, yp = [], []
    for x, y in te:
        yp.extend(tf.argmax(student(x, training=False), 1).numpy())
        yt.extend(y.numpy())
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    del yt, yp
    return acc, f1


def run_cv_pair(paths, labels, teacher_path, seed):
    """1 seed → 5-fold CV, mỗi fold train CẢ no-KD lẫn KD trên CÙNG split."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    nokd_acc, nokd_f1, kd_acc, kd_f1 = [], [], [], []

    for fold_idx, (trainval_idx, te_idx) in enumerate(skf.split(paths, labels), 1):
        print(f"  Fold {fold_idx}/{N_FOLDS}")
        
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        tr_sub, val_sub = next(inner.split(paths[trainval_idx], labels[trainval_idx]))
        tr_idx, val_idx = trainval_idx[tr_sub], trainval_idx[val_sub]
        
        # ── PHẦN 1: Train no-KD ──
        try:
            # [Yêu cầu 5] Tạo dataset và tính class weight dành riêng cho no-KD
            tr = make_ds(paths[tr_idx], labels[tr_idx], True, seed)
            va = make_ds(paths[val_idx], labels[val_idx], False, seed)
            te = make_ds(paths[te_idx], labels[te_idx], False, seed)
            cw = cw_from(labels[tr_idx])

            s_nokd = build_student()
            s_nokd.compile(
                optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])
            es1 = tf.keras.callbacks.EarlyStopping(
                patience=6, restore_best_weights=True,
                monitor="val_accuracy", mode="max")
            
            s_nokd.fit(tr, validation_data=va, epochs=STUDENT_EPOCHS_CV,
                       class_weight=cw, callbacks=[es1], verbose=0)
            a, f = _eval(s_nokd, te)
            nokd_acc.append(a); nokd_f1.append(f)
            
        finally:
            if 's_nokd' in locals(): del s_nokd
            if 'es1' in locals(): del es1
            # [Yêu cầu 5] Dọn sạch object dataset của no-KD
            if 'tr' in locals(): del tr
            if 'va' in locals(): del va
            if 'te' in locals(): del te
            
            tf.keras.backend.clear_session()
            gc.collect()

        # ── PHẦN 2: Train KD ──
        try:
            # [Yêu cầu 5] Tái tạo lại (recreate) dataset hoàn toàn mới cho tiến trình KD
            tr = make_ds(paths[tr_idx], labels[tr_idx], True, seed)
            va = make_ds(paths[val_idx], labels[val_idx], False, seed)
            te = make_ds(paths[te_idx], labels[te_idx], False, seed)

            teacher = tf.keras.models.load_model(teacher_path, compile=False)
            teacher.trainable = False
            
            s_kd = build_student()
            distiller = Distiller(s_kd, teacher)
            distiller.compile(
                optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                alpha=C.KD_ALPHA, temperature=C.KD_TEMPERATURE)
            es2 = tf.keras.callbacks.EarlyStopping(
                patience=6, restore_best_weights=True,
                monitor="val_accuracy", mode="max")
            
            distiller.fit(tr, validation_data=va, epochs=STUDENT_EPOCHS_CV,
                          class_weight=cw, callbacks=[es2], verbose=0)
            a, f = _eval(s_kd, te)
            kd_acc.append(a); kd_f1.append(f)

            # [Yêu cầu 6] Xóa thủ công teacher ngay lập tức sau khi eval xong
            del teacher

        finally:
            # Kiểm tra 'teacher' in locals() đảm bảo an toàn kể cả khi try block lỗi giữa chừng
            if 'teacher' in locals(): del teacher
            if 's_kd' in locals(): del s_kd
            if 'distiller' in locals(): del distiller
            if 'es2' in locals(): del es2
            # [Yêu cầu 5] Dọn sạch object dataset của KD
            if 'tr' in locals(): del tr
            if 'va' in locals(): del va
            if 'te' in locals(): del te
            
            tf.keras.backend.clear_session()
            gc.collect()

    return (np.mean(nokd_acc), np.mean(nokd_f1),
            np.mean(kd_acc), np.mean(kd_f1))


def main():
    print("=" * 60)
    print("no-KD vs KD — 5-fold CV, multi-seed (head-to-head)")
    print("=" * 60)
    print(f"Dataset : {C.DATASET_DIR}")
    print(f"Teacher : {C.TEACHER_PATH}")
    print(f"Batch   : {C.BATCH_SIZE}  |  KD: T={C.KD_TEMPERATURE}, α={C.KD_ALPHA}")
    print(f"Seeds   : {SEEDS}")

    if not C.TEACHER_PATH.exists():
        print(f"\n❌ Không thấy teacher: {C.TEACHER_PATH} — train teacher trước.")
        return

    paths, labels = gather_all()
    print(f"Tổng ảnh: {len(paths)}\n")

    nokd_list, kd_list, gain_list = [], [], []
    
    # [Yêu cầu 7] Khởi tạo môi trường hoàn toàn sạch trước khi bắt đầu seed đầu tiên
    tf.keras.backend.clear_session()
    gc.collect()
    
    for seed in SEEDS:
        print(f"{'─'*60}\nSEED {seed}\n{'─'*60}")
        n_acc, n_f1, k_acc, k_f1 = run_cv_pair(paths, labels, C.TEACHER_PATH, seed)
        gain = (k_acc - n_acc) * 100
        nokd_list.append(n_acc); kd_list.append(k_acc); gain_list.append(gain)
        print(f"  no-KD: acc={n_acc:.4f} f1={n_f1:.4f}  |  "
              f"KD: acc={k_acc:.4f} f1={k_f1:.4f}  |  gain={gain:+.2f}pp")

    nokd = np.array(nokd_list); kd = np.array(kd_list); gain = np.array(gain_list)
    print(f"\n{'='*60}\nKẾT QUẢ CV — no-KD vs KD ({len(SEEDS)} seeds)\n{'='*60}")
    print("Per-seed:")
    for s, n, k, g in zip(SEEDS, nokd, kd, gain):
        print(f"  seed {s:>4}:  no-KD={n:.4f}  KD={k:.4f}  gain={g:+.2f}pp")
    print("-" * 60)
    print(f"no-KD : {nokd.mean():.4f} ± {nokd.std():.4f}")
    print(f"KD    : {kd.mean():.4f} ± {kd.std():.4f}")
    print(f"gain  : {gain.mean():+.2f}pp ± {gain.std():.2f}pp")
    print("=" * 60)
    print("\n→ Đọc kết quả:")
    print("  - Nếu gain.mean > 0 và > gain.std → KD giúp ổn định. Báo cáo con số.")
    print("  - Nếu gain.mean ≈ 0 hoặc âm, hoặc std nuốt mean → KD KHÔNG cho")
    print("    cải thiện đáng tin trên dataset này. Báo cáo ĐÚNG như vậy — đó là")
    print("    phát hiện hợp lệ (teacher–student gap hẹp), không phải thất bại.")


if __name__ == "__main__":
    main()