#!/usr/bin/env python3
"""
config.py — Cấu hình trung tâm cho pipeline Knowledge Distillation.

Cấu hình chốt sau thực nghiệm:
  - Teacher: EfficientNet-B0 (test 0.92, CV 92.62% ± 3.99%)
  - Student: MobileNetV3-Large (Small quá nhỏ → 75.9%; Large → 88.5%)
  - Augmentation NHẸ (brightness/contrast) — aug hình học làm tụt accuracy
  - Class weights để xử lý imbalance (ok 345 vs defect 40-52)
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
DATASET_DIR = Path("dataset_v4")
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Classes (7 defect types) ───────────────────────────────
CLASS_NAMES = ["bridge", "missing", "ok", "rotate",
               "shift", "tilt", "tombstone"]
NUM_CLASSES = len(CLASS_NAMES)

# Class nhạy góc — KHÔNG áp angle augmentation cho các class này
# (xoay/lật "rotate" → trông như "ok", phá nhãn)
ANGLE_SENSITIVE = ["rotate", "shift", "tilt", "tombstone"]

# ── Image & Batch ──────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
SEED       = 42

# ── Augmentation toggles ───────────────────────────────────
# USE_AUGMENTATION: bật/tắt toàn bộ augmentation
#   True  → augment brightness/contrast (+ angle nếu USE_ANGLE_AUG=True)
#   False → không augment gì cả
USE_AUGMENTATION = True

# USE_ANGLE_AUG: bật/tắt flip + rotation (CHỈ cho ok/missing/bridge)
#   True  → thêm flip + rotate±45° cho class không nhạy góc
#   False → chỉ brightness/contrast (cấu hình đã test, CV 92.62%)
USE_ANGLE_AUG = False       # ← best config: 90.8% acc, F1=0.837

# ── Training ───────────────────────────────────────────────
TEACHER_EPOCHS = 30
STUDENT_EPOCHS = 40
LEARNING_RATE  = 1e-3       # giai đoạn 1: train head
FINE_TUNE_LR   = 1e-5       # giai đoạn 2: fine-tune backbone

# ── Knowledge Distillation ─────────────────────────────────
# Giá trị tối ưu từ hyperparameter sweep (T=4.0, α=0.5 → acc 0.851, F1 0.745)
KD_TEMPERATURE = 4.0        # nhiệt độ làm mềm soft label (Hinton 2015)
KD_ALPHA       = 0.5        # 0.5 soft loss + 0.5 hard loss (tốt nhất từ sweep)

# ── Model Paths ────────────────────────────────────────────
TEACHER_PATH        = OUTPUT_DIR / "teacher_effnetb0_v4.keras"
STUDENT_KD_PATH     = OUTPUT_DIR / "student_mobilenetv3_kd_v4.keras"
STUDENT_NOKD_PATH   = OUTPUT_DIR / "student_mobilenetv3_nokd_v4.keras"
TFLITE_FLOAT32_PATH = OUTPUT_DIR / "student_float32.tflite"   # ← thêm
TFLITE_PATH         = OUTPUT_DIR / "student_int8.tflite"