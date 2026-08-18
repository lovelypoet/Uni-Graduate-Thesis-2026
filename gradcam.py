#!/usr/bin/env python3
"""
gradcam.py — Grad-CAM: chứng minh model nhìn ĐÚNG vùng lỗi.

Tạo heatmap chồng lên ảnh để thấy model "chú ý" vào đâu.
Lấy 1 ảnh mỗi class từ test set, xuất grid 2 hàng (gốc + heatmap).

Chạy:
  python3 gradcam.py               # student KD
  python3 gradcam.py teacher       # teacher
"""
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import config as C


def find_last_conv_layer(model):
    """Tìm layer 4D (conv feature map) cuối cùng trong model."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("Không tìm thấy conv layer 4D")


def make_gradcam_heatmap(img_array, model, last_conv_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            raise ValueError("no grad")
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), int(pred_index)
    except Exception:
        # fallback: activation map trung bình (vẫn cho thấy vùng kích hoạt)
        sub = tf.keras.models.Model(model.inputs,
                                    model.get_layer(last_conv_name).output)
        conv_out = sub(img_array)[0]
        preds = model(img_array)
        pi = int(tf.argmax(preds[0])) if pred_index is None else pred_index
        hm = tf.reduce_mean(conv_out, axis=-1).numpy()
        hm = np.maximum(hm, 0) / (hm.max() + 1e-8)
        return hm, pi


def overlay_heatmap(img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_hm = jet[heatmap]
    jet_hm = tf.image.resize(jet_hm[np.newaxis],
                             (C.IMG_SIZE, C.IMG_SIZE))[0].numpy()
    return np.clip(0.6 * (img / 255.0) + 0.4 * jet_hm, 0, 1)


def main(which="student"):
    path = C.TEACHER_PATH if which == "teacher" else C.STUDENT_KD_PATH
    print(f"Grad-CAM cho: {path}")
    model = tf.keras.models.load_model(path)
    last_conv = find_last_conv_layer(model)
    print(f"Conv layer cuối: {last_conv}")

    fig, axes = plt.subplots(2, C.NUM_CLASSES,
                             figsize=(3 * C.NUM_CLASSES, 6))
    for idx, cls in enumerate(C.CLASS_NAMES):
        d = C.DATASET_DIR / "test" / cls
        files = list(d.glob("*.jpg"))
        if not files:
            axes[0, idx].axis("off"); axes[1, idx].axis("off")
            continue
        img = tf.keras.utils.load_img(
            files[0], target_size=(C.IMG_SIZE, C.IMG_SIZE))
        arr = tf.keras.utils.img_to_array(img)[np.newaxis]

        hm, pred_idx = make_gradcam_heatmap(arr, model, last_conv)
        pred_cls = C.CLASS_NAMES[pred_idx]

        axes[0, idx].imshow(arr[0].astype("uint8"))
        ok = "✓" if pred_cls == cls else "✗"
        axes[0, idx].set_title(f"{cls}\n→ {pred_cls} {ok}", fontsize=9)
        axes[0, idx].axis("off")
        axes[1, idx].imshow(overlay_heatmap(arr[0], hm))
        axes[1, idx].axis("off")

    fig.suptitle(f"Grad-CAM — {which} (hàng trên: ảnh gốc, "
                 f"dưới: vùng model chú ý)", fontsize=13)
    fig.tight_layout()
    out = C.OUTPUT_DIR / f"gradcam_{which}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"💾 Đã lưu: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "student")