#!/usr/bin/env python3
"""
generate_misclassified_grid.py — Tạo lưới ảnh sai + Grad-CAM cho
Section 4.4.3 "Misclassified Examples" của thesis (Figure~\\ref{fig:misclassified_grid}).

Làm gì:
  1. Chạy model (mặc định: student KD) trên TOÀN BỘ test split của
     DATASET_DIR hiện tại trong config.py (đặt = dataset_v4 cho đúng
     dataset dùng trong Chapter 4).
  2. Tìm các ảnh bị đoán sai, ưu tiên đúng 3 cặp nhầm lẫn đã nêu trong
     thesis (Section 4.4.1 Confusion Matrix): rotate/ok, shift/ok,
     bridge/tombstone — nghĩa là: true=rotate & pred=ok (và ngược lại),
     true=shift & pred=ok (và ngược lại), true=bridge & pred=tombstone
     (và ngược lại).
  3. Với mỗi ảnh chọn, tính Grad-CAM (gradient thật, không phải bản
     approximation trên app) trên layer conv cuối cùng của backbone.
  4. Ghép thành 1 lưới ảnh, mỗi ô có: ảnh gốc + overlay CAM, caption
     "true: X, pred: Y, conf: Z%".
  5. Lưu ra outputs/misclassified_grid.png — chèn trực tiếp vào
     \\includegraphics trong thesis, thay cho khung
     \\textcolor{accentred}{[Insert...]}.

Chạy:
    python3 generate_misclassified_grid.py                # dùng student KD
    python3 generate_misclassified_grid.py --model teacher # dùng teacher
    python3 generate_misclassified_grid.py --max-per-pair 3
"""
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

# Cặp nhầm lẫn ưu tiên, đúng theo Section 4.4.1 của thesis.
# Mỗi tuple là (true_class, pred_class); cả hai chiều của mỗi cặp đều
# được tìm (vd rotate->ok VÀ ok->rotate).
PRIORITY_PAIRS = [
    ("rotate", "ok"), ("ok", "rotate"),
    ("shift", "ok"), ("ok", "shift"),
    ("bridge", "tombstone"), ("tombstone", "bridge"),
]


def load_test_images():
    """Load test split kèm đường dẫn gốc (để hiển thị ảnh thật, không
    qua augmentation)."""
    paths, labels = [], []
    test_dir = C.DATASET_DIR / "test"
    for idx, cls in enumerate(C.CLASS_NAMES):
        d = test_dir / cls
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jpg")):
            paths.append(str(f))
            labels.append(idx)
    return paths, np.array(labels)


def preprocess_image(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
    return img.numpy()


def find_last_conv_layer(model):
    """Tìm layer Conv2D cuối cùng trong model (dùng cho Grad-CAM).
    Duyệt ngược để bắt cả trường hợp backbone lồng trong 1 sub-model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # backbone thường là 1 Model con (Functional) lồng bên trong
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("Không tìm thấy Conv2D layer nào trong model.")


def compute_gradcam(model, img_array, class_idx, last_conv_layer_name):
    """Grad-CAM chuẩn (Selvaraju et al. 2017): gradient thật của logit
    lớp dự đoán theo activation map của layer conv cuối."""
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
        if last_conv_layer_name in [l.name for l in model.layers]
        else [_find_nested_layer(model, last_conv_layer_name).output, model.output]
    )
    img_batch = np.expand_dims(img_array, axis=0)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _find_nested_layer(model, name):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                return layer.get_layer(name)
            except ValueError:
                continue
    raise ValueError(f"Layer {name} không tìm thấy kể cả trong sub-model.")


def overlay_heatmap(img, heatmap, alpha=0.45):
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis],
                                       (img.shape[0], img.shape[1])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)
    cmap = matplotlib.colormaps["jet"]
    heatmap_color = cmap(heatmap_resized)[..., :3] * 255
    overlay = img * (1 - alpha) + heatmap_color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def select_examples(paths, labels, y_pred, y_conf, max_per_pair):
    """Chọn ảnh sai ưu tiên theo PRIORITY_PAIRS; nếu không đủ ảnh cho
    các cặp ưu tiên, bổ sung thêm các lỗi khác để lấp đầy lưới."""
    name_to_idx = {c: i for i, c in enumerate(C.CLASS_NAMES)}
    selected = []
    used = set()

    for true_name, pred_name in PRIORITY_PAIRS:
        t_idx, p_idx = name_to_idx[true_name], name_to_idx[pred_name]
        matches = [i for i in range(len(paths))
                   if labels[i] == t_idx and y_pred[i] == p_idx and i not in used]
        for i in matches[:max_per_pair]:
            selected.append(i)
            used.add(i)

    if len(selected) < 6:
        other_wrong = [i for i in range(len(paths))
                        if labels[i] != y_pred[i] and i not in used]
        for i in other_wrong[: (6 - len(selected))]:
            selected.append(i)
            used.add(i)

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["student", "teacher"], default="student")
    parser.add_argument("--max-per-pair", type=int, default=2)
    args = parser.parse_args()

    model_path = C.TEACHER_PATH if args.model == "teacher" else C.STUDENT_KD_PATH
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    last_conv = find_last_conv_layer(model)
    print(f"Grad-CAM sẽ dùng layer conv cuối: {last_conv}")

    paths, labels = load_test_images()
    print(f"Test set: {len(paths)} ảnh (từ {C.DATASET_DIR}/test)")

    imgs = np.stack([preprocess_image(p) for p in paths])
    logits = model.predict(imgs, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()
    y_pred = np.argmax(probs, axis=1)
    y_conf = np.max(probs, axis=1)

    acc = np.mean(y_pred == labels)
    print(f"Accuracy trên test set này: {acc:.4f} "
          f"(so với số trong thesis để xác nhận đúng model/dataset)")

    selected = select_examples(paths, labels, y_pred, y_conf, args.max_per_pair)
    if not selected:
        print("❌ Không tìm thấy ảnh sai nào — model đoán đúng hết? Kiểm tra lại dataset/model.")
        return
    print(f"Chọn {len(selected)} ảnh sai để đưa vào lưới.")

    n = len(selected)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(selected):
        img = imgs[idx]
        heatmap = compute_gradcam(model, img, y_pred[idx], last_conv)
        overlay = overlay_heatmap(img, heatmap)
        ax = axes[ax_i]
        ax.imshow(overlay.astype(np.uint8))
        ax.axis("off")
        true_name = C.CLASS_NAMES[labels[idx]]
        pred_name = C.CLASS_NAMES[y_pred[idx]]
        ax.set_title(
            f"true: {true_name}\npred: {pred_name}  ({y_conf[idx]*100:.0f}%)",
            fontsize=10)

    for ax_i in range(len(selected), len(axes)):
        axes[ax_i].axis("off")

    plt.tight_layout()
    out_path = C.OUTPUT_DIR / "misclassified_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Đã lưu: {out_path}")
    print("→ Copy file này vào thư mục figures/ cạnh thesis_final.tex,")
    print("  rồi thay khung \\textcolor{accentred}{[Insert a grid...]}")
    print("  trong Section 4.4.3 bằng:")
    print(r"  \includegraphics[width=0.85\textwidth]{figures/misclassified_grid.png}")


if __name__ == "__main__":
    main()