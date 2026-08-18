#!/usr/bin/env python3
"""
benchmark.py — Bảng ablation: teacher / student / TFLite.

Đo 4 metrics mỗi model: Accuracy, F1-macro, Size (MB), Latency (ms).
Output → bảng ablation chính của thesis (Chapter IV).

Lưu ý: latency đo trên máy local. Latency thật trên S10e (Snapdragon 855)
đo bằng Android app. Ghi rõ hardware khi báo cáo.

Chạy: python3 benchmark.py
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import OpResolverType
from sklearn.metrics import accuracy_score, f1_score
import config as C


def load_test_numpy():
    ds = tf.keras.utils.image_dataset_from_directory(
        C.DATASET_DIR / "test", labels="inferred", label_mode="int",
        class_names=C.CLASS_NAMES, image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=C.BATCH_SIZE, shuffle=False)
    X, y = [], []
    for img, lab in ds:
        X.append(img.numpy()); y.append(lab.numpy())
    return np.concatenate(X), np.concatenate(y)


def eval_keras(path, X, y):
    model = tf.keras.models.load_model(path)
    _ = model(X[:2], training=False)  # warm-up
    preds = []
    t0 = time.perf_counter()
    for i in range(len(X)):
        logits = model(X[i:i+1], training=False)
        preds.append(int(tf.argmax(logits, axis=1)[0]))
    latency_ms = (time.perf_counter() - t0) / len(X) * 1000
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return acc, f1, path.stat().st_size / 1e6, latency_ms


def eval_tflite(path, X, y):
    """Evaluate TFLite. TẮT XNNPACK (BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    để tránh lỗi delegate với INT8 + float fallback."""
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    assert inp["dtype"] == np.float32, f"Input dtype {inp['dtype']} != float32"

    dummy = X[:1].astype(np.float32)
    interpreter.set_tensor(inp["index"], dummy)
    interpreter.invoke()  # warm-up

    preds = []
    t_total = 0.0
    for i in range(len(X)):
        x = X[i:i+1].astype(np.float32)
        interpreter.set_tensor(inp["index"], x)
        t0 = time.perf_counter()
        interpreter.invoke()
        t_total += time.perf_counter() - t0
        preds.append(int(np.argmax(interpreter.get_tensor(out["index"]))))

    latency_ms = t_total / len(X) * 1000
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return acc, f1, path.stat().st_size / 1e6, latency_ms


def main():
    print("=" * 70)
    print("BENCHMARK — Ablation Study")
    print("=" * 70)
    print("\n📂 Loading test set...")
    X, y = load_test_numpy()
    print(f"   {len(X)} images, {C.NUM_CLASSES} classes")

    rows = []
    if C.TEACHER_PATH.exists():
        print("\n🔍 Teacher (EfficientNet-B0)...")
        rows.append(("Teacher (EffNetB0)", *eval_keras(C.TEACHER_PATH, X, y)))
    if C.STUDENT_NOKD_PATH.exists():
        print("🔍 Student (no KD)...")
        rows.append(("Student (no KD)", *eval_keras(C.STUDENT_NOKD_PATH, X, y)))
    if C.STUDENT_KD_PATH.exists():
        print("🔍 Student (KD)...")
        rows.append(("Student (KD)", *eval_keras(C.STUDENT_KD_PATH, X, y)))
    float32_path = C.OUTPUT_DIR / "student_float32.tflite"
    if float32_path.exists():
        print("🔍 TFLite float32...")
        rows.append(("TFLite float32", *eval_tflite(float32_path, X, y)))
    if C.TFLITE_PATH.exists():
        print("🔍 TFLite INT8...")
        rows.append(("TFLite INT8", *eval_tflite(C.TFLITE_PATH, X, y)))

    print("\n" + "=" * 72)
    print(f"  {'Model':<22} {'Accuracy':>9} {'F1-macro':>9} {'Size(MB)':>9} {'Latency(ms)':>12}")
    print("  " + "-" * 68)
    for name, acc, f1, size, lat in rows:
        print(f"  {name:<22} {acc:>9.4f} {f1:>9.4f} {size:>9.2f} {lat:>12.2f}")
    print("=" * 72)

    if len(rows) >= 2:
        teacher_acc = rows[0][1]
        print("\n📊 Accuracy drop vs Teacher:")
        for name, acc, f1, size, lat in rows[1:]:
            print(f"   {name:<22}: {(teacher_acc - acc)*100:+.2f}%")

    print("\n⚠️  Latency local machine — ghi rõ hardware khi báo cáo.")
    print("   Latency thật trên S10e (Snapdragon 855) đo bằng Android app.")


if __name__ == "__main__":
    main()