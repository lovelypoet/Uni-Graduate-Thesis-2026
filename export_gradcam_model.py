#!/usr/bin/env python3
"""
export_gradcam_model.py — Export student model với 2 đầu ra cho Grad-CAM realtime.

Model thường: input → logits (7)
Model này:    input → [logits (7), feature_maps (7×7×C)]
              → để tính CAM heatmap trên device (không cần gradient)

Chạy: python3 export_gradcam_model.py
Output: outputs/student_gradcam_v4.tflite
"""
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import config as C

_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass


def find_last_conv_layer(model):
    """Tìm conv layer cuối (feature map cho CAM)."""
    for layer in reversed(model.layers):
        # tìm layer có output 4D (B, H, W, C)
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                print(f"  Last conv-like layer: {layer.name} {shape}")
                return layer
        except Exception:
            continue
    return None


def main(model_path=None, out_name="student_gradcam_v4.tflite"):
    print("="*60)
    print("EXPORT GRAD-CAM MODEL (2 outputs: logits + features)")
    print("="*60)

    path = model_path if model_path else C.STUDENT_KD_PATH
    print(f"\n📂 Loading student: {path}")
    model = tf.keras.models.load_model(path)

    # Tìm feature map layer cuối
    print("\n🔍 Finding last conv layer...")
    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        print("❌ Không tìm thấy conv layer 4D")
        return

    # Tạo model 2 đầu ra: [logits, feature_maps]
    print("\n🔧 Building 2-output model...")
    grad_model = models.Model(
        inputs=model.input,
        outputs=[model.output, last_conv.output],
        name="student_gradcam"
    )
    print(f"   Output 1 (logits):   {model.output.shape}")
    print(f"   Output 2 (features): {last_conv.output.shape}")

    # Convert sang TFLite (float32 — ổn định cho Android)
    print("\n🔄 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(grad_model)
    # giữ float32 cho ổn định + có feature map chính xác
    tflite_model = converter.convert()

    out_path = C.OUTPUT_DIR / out_name
    out_path.write_bytes(tflite_model)
    print(f"   ✅ Saved: {out_path}")
    print(f"   Size: {out_path.stat().st_size/1e6:.2f} MB")

    # Verify
    print("\n🔍 Verifying outputs...")
    interp = tf.lite.Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    out_details = interp.get_output_details()
    print(f"   Number of outputs: {len(out_details)}")
    for i, o in enumerate(out_details):
        print(f"   Output {i}: shape={o['shape']} dtype={o['dtype'].__name__}")

    print("\n📱 Copy student_gradcam.tflite → Android assets/")
    print("   App tự nhận diện output theo SHAPE (2D=logits, 4D=features) — TFLite co the dao thu tu")


if __name__ == "__main__":
    # Cách dùng:
    #   python3 export_gradcam_model.py                    → student_kd.keras → student_gradcam_v4.tflite
    #   python3 export_gradcam_model.py v2b0               → student_v2b0_kd.keras → student_v2b0_gradcam.tflite
    if len(sys.argv) > 1 and sys.argv[1] == "v2b0":
        main(model_path=C.OUTPUT_DIR / "student_v2b0_kd.keras",
             out_name="student_v2b0_gradcam.tflite")
    else:
        main()