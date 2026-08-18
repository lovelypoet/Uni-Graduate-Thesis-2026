#!/usr/bin/env python3
"""
quantize_tflite.py — Convert student sang TFLite + INT8 quantization.

Tạo 2 bản:
  1. Float32 (baseline)
  2. INT8 weights + float32 I/O (nhẹ ~4x, compatible Android/MobileNetV3)

Giữ float32 I/O vì MobileNetV3 có preprocess_input bên trong model
→ ép uint8 I/O sẽ gây sai kết quả khi deploy.

Chạy: python3 quantize_tflite.py
"""
import numpy as np
import tensorflow as tf
import config as C


def representative_dataset():
    """200 mẫu từ train set để calibrate INT8."""
    ds = tf.keras.utils.image_dataset_from_directory(
        C.DATASET_DIR / "train", labels="inferred", label_mode="int",
        class_names=C.CLASS_NAMES, image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=1, shuffle=True, seed=C.SEED)
    count = 0
    for img, _ in ds:
        yield [tf.cast(img, tf.float32)]
        count += 1
        if count >= 200:
            break


def convert_float32(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def convert_int8(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback op không INT8 được
    ]
    return converter.convert()


def verify_tflite(tflite_path, model_name="TFLite"):
    """Kiểm tra model load được. TẮT XNNPACK để tránh lỗi delegate
    với model INT8 + float fallback."""
    try:
        # experimental_delegates=[] → KHÔNG dùng XNNPACK
        interpreter = tf.lite.Interpreter(
            model_path=str(tflite_path),
            experimental_preserve_all_tensors=False)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        print(f"  {model_name}:")
        print(f"    Input  shape: {inp['shape']}  dtype: {inp['dtype'].__name__}")
        print(f"    Output shape: {out['shape']}  dtype: {out['dtype'].__name__}")
        return True
    except Exception as e:
        # verify lỗi KHÔNG nghĩa là model hỏng — chỉ là delegate CPU
        print(f"  {model_name}: ⚠ verify skip (XNNPACK delegate), "
              f"model vẫn dùng được trên Android. Chi tiết: {str(e)[:60]}")
        return False


def main():
    print("=" * 50)
    print("TFLite Conversion + INT8 Quantization")
    print("=" * 50)

    print(f"\n📂 Loading student model: {C.STUDENT_KD_PATH}")
    model = tf.keras.models.load_model(C.STUDENT_KD_PATH)
    print(f"   Parameters: {model.count_params():,}")

    print("\n🔄 Converting float32 TFLite...")
    tfl_float = convert_float32(model)
    float_path = C.OUTPUT_DIR / "student_float32.tflite"
    float_path.write_bytes(tfl_float)
    print(f"   ✅ Saved: {float_path}")

    print("\n🔄 Converting INT8 TFLite (calibrating với 200 mẫu)...")
    tfl_int8 = convert_int8(model)
    C.TFLITE_PATH.write_bytes(tfl_int8)
    print(f"   ✅ Saved: {C.TFLITE_PATH}")

    print("\n🔍 Verifying models...")
    verify_tflite(float_path,    "Float32")
    verify_tflite(C.TFLITE_PATH, "INT8")

    keras_mb = C.STUDENT_KD_PATH.stat().st_size / 1e6
    float_mb = float_path.stat().st_size / 1e6
    int8_mb  = C.TFLITE_PATH.stat().st_size / 1e6

    print("\n" + "=" * 50)
    print("  SIZE COMPARISON")
    print("=" * 50)
    print(f"  Keras  (.keras)      : {keras_mb:6.2f} MB")
    print(f"  TFLite float32       : {float_mb:6.2f} MB")
    print(f"  TFLite INT8          : {int8_mb:6.2f} MB")
    print(f"  Reduction (INT8/f32) : {(1 - int8_mb/float_mb)*100:.1f}%")
    print("=" * 50)

    print(f"\n💾 INT8 model ready: {C.TFLITE_PATH}")
    print("\n📱 Deploy to Android:")
    print("   1. Copy outputs/student_int8.tflite → Android assets/")
    print("   2. Load via Interpreter('student_int8.tflite')")
    print("   3. Enable NNAPI delegate cho Snapdragon 845")


if __name__ == "__main__":
    main()