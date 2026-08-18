#!/usr/bin/env python3
"""
evaluate.py — Đánh giá chi tiết 1 model trên test set.
In: accuracy, F1-macro, classification report, confusion matrix (lưu PNG).

Chạy:
  python3 evaluate.py outputs/teacher_effnetb0.keras
  python3 evaluate.py outputs/student_mobilenetv3_kd.keras
"""
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)
import config as C


def load_test():
    return tf.keras.utils.image_dataset_from_directory(
        C.DATASET_DIR / "test",
        labels="inferred", label_mode="int",
        class_names=C.CLASS_NAMES,
        image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=C.BATCH_SIZE, shuffle=False,
    )


def main(model_path):
    print(f"Đánh giá: {model_path}")
    model = tf.keras.models.load_model(model_path)
    test = load_test()

    y_true, y_pred = [], []
    for x, y in test:
        logits = model(x, training=False)
        y_pred.extend(tf.argmax(logits, axis=1).numpy())
        y_true.extend(y.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"\n🎯 Accuracy : {acc:.4f}")
    print(f"🎯 F1-macro : {f1m:.4f}\n")

    print("Báo cáo chi tiết theo class:")
    print(classification_report(y_true, y_pred,
          target_names=C.CLASS_NAMES, digits=3, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(C.NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(C.NUM_CLASSES)); ax.set_yticks(range(C.NUM_CLASSES))
    ax.set_xticklabels(C.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(C.CLASS_NAMES)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion Matrix (acc={acc:.3f}, F1={f1m:.3f})")
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im); fig.tight_layout()
    out = C.OUTPUT_DIR / (model_path.split("/")[-1].replace(".keras", "_cm.png"))
    fig.savefig(out, dpi=120)
    print(f"\n💾 Confusion matrix: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng: python3 evaluate.py <đường_dẫn_model.keras>")
        sys.exit(1)
    main(sys.argv[1])