<p align="center">
  <img src="assets/banner.png" alt="PCB Defect Inspector" width="520" />
</p>

<h2 align="center">Knowledge Distillation for PCB Defect Classification</h2>

---

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-3f464e?style=flat-square" alt="license" />
  <img src="https://img.shields.io/badge/python-3.12+-2d5a4c?style=flat-square" alt="python" />
  <img src="https://img.shields.io/badge/kotlin-1.9+-purple?style=flat-square" alt="kotlin" />
  <img src="https://img.shields.io/badge/android-minSdk%2026-3DDC84?style=flat-square" alt="android" />
  <img src="https://img.shields.io/badge/TensorFlow%20Lite-2.x-FF6F00?style=flat-square" alt="tflite" />
  <img src="https://img.shields.io/badge/dataset-PCB%20Component-blue?style=flat-square" alt="dataset" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="prs welcome" />
</p>

## Overview

Post-AOI (Automated Optical Inspection) on Surface Mount Technology (SMT) production lines typically relies on rigid rule-based heuristics tied to board CAD data or heavy GPU-hosted deep learning models. Neither option can be carried to a manual rework bench or a disconnected small-scale line.

This project compresses an EfficientNet-B0 teacher network into a lightweight MobileNetV3-Large student via Knowledge Distillation (KD), exporting the distilled model to TensorFlow Lite and deploying it inside a real-time Android application (**PCB Inspector**) with live camera inference and on-device visual explanations.

**Đặc điểm nổi bật của bộ dữ liệu:** Đây là bộ dữ liệu tôi tự thu dựa trên kinh nghiệm làm việc trong nhà máy, một trong những bộ dataset cung cấp đầy đủ hình ảnh lỗi PCB — nhưng không phải là lỗi mạch in (board-level defects), mà là lỗi định vị linh kiện trên PCB (component placement defects), bao gồm thiếu linh kiện, lệch vị trí, nghiêng, tombstone, xoay và bridge.

**Dataset highlight:** This is a dataset I collected myself based on my experience working in a factory — one of the rare repositories worldwide that provides comprehensive PCB defect images — not board-level circuit defects, but **component placement defects on PCBs**, including missing, shift, tilt, tombstone, rotate, and bridge defects.

## Key Highlights

- **CAD-Free 7-Class Dataset (v4 — 979 images):** Curated from real passive two-terminal SMT components across 7 classes (`ok`, `missing`, `shift`, `tilt`, `tombstone`, `rotate`, `bridge`) with a custom browser-based annotation tool (**PCB Component Labeler**).
- **Domain-Specific Augmentation Finding:** Geometric augmentations (flip, rotation, zoom) degrade accuracy by 8.9 pp due to spatial label corruption, whereas light optical augmentation (±10% brightness/contrast) improves generalization.
- **Effective Model Compression:** Reduced model footprint by 4.4× (52.86 MB → 11.94 MB in float32 TFLite) while retaining 97.5% of the teacher's test accuracy (86.2% vs. 88.4%) and more than doubling recall on the hardest `rotate` class (0.333 → 0.722).
- **Real-Time Mobile Deployment:** Achieved 17.3 ms inference latency (~57.7 FPS) on a Samsung Galaxy S22+ and 18.1 ms (~55.2 FPS) on a 2019 Samsung Galaxy S10e with Android NNAPI enabled.

## System Pipeline

```text
[PCB Component Crops (224x224 RGB)]
         │
         ▼
┌────────────────────────────────────────┐
│   Teacher: EfficientNet-B0 (Frozen)    │ ──► Soft Labels (T = 4.0)
└────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────┐    ┌──────────────────┐
│   Student: MobileNetV3-Large           │◄───│  KD Loss (α=0.5) │
└────────────────────────────────────────┘    └──────────────────┘
         │
         ▼
[TFLite Export (Two-Output: Logits + 7×7 Feature Map)]
         │
         ▼
[Android App: CameraX + NNAPI + Real-Time CAM Overlay]
```

## Benchmark Results (Dataset v4)

### Model Performance & Compression

| Model / Export Format | Test Acc | F1-Macro | Params | FLOPs | Size | Compression |
|-----------------------|----------|----------|--------|-------|------|-------------|
| Teacher (EfficientNet-B0, Keras) | 88.4% | 0.865 | 4.38M | 0.801G | 52.86 MB | 1.0× |
| Student Baseline (MobileNetV3-L, no-KD) | 83.3% | 0.803 | 3.00M | 0.444G | 36.65 MB | 1.4× |
| Student Distilled (MobileNetV3-L, KD) | 86.2% | 0.840 | 3.00M | 0.444G | 12.69 MB | 4.2× |
| Deployed TFLite (float32) | 86.2% | 0.840 | 3.00M | 0.444G | 11.94 MB | 4.4× |
| Quantised TFLite (INT8 PTQ) (benchmarked) | 73.9% | 0.678 | 3.00M | 0.444G | 3.52 MB | 15.0× |

*Note:* float32 was chosen for production deployment to preserve the fidelity of the 7×7 final convolutional feature maps used for on-device CAM heatmaps.

### On-Device Mobile Latency (50-Inference Average)

| Device | SoC (Year) | NNAPI | Mean Latency | Throughput |
|--------|------------|-------|--------------|------------|
| Samsung Galaxy S10e | Snapdragon 855 (2019) | ON | 18.1 ms | 55.2 FPS |
| Samsung Galaxy S10e | Snapdragon 855 (2019) | OFF | 25.9 ms | 38.6 FPS |
| Samsung Galaxy S22+ | Snapdragon 8 Gen 1 (2022) | ON | 17.3 ms | 57.7 FPS |
| Samsung Galaxy S22+ | Snapdragon 8 Gen 1 (2022) | OFF | 17.9 ms | 55.9 FPS |

## Android Application Features (PCB Inspector)

Built in Kotlin (minSdk 26) using CameraX and TensorFlow Lite:

- **Live Camera Inference:** Google-Translate-style continuous component inspection with real-time class prediction and confidence scores.
- **On-Device Visual Explanation:** Activation-weighted Class Activation Mapping (CAM) jet-colour heatmap overlay highlighting defect regions (7×7 feature map).
- **Field & Evaluation Controls:** Pinch-to-zoom for tight single-component framing, live NNAPI ON/OFF delegate toggle, frame capture with heatmap overlay, and an automated 50-run benchmark mode.

<p align="center">
  <img src="assets/Screenshot%202026-11-20%20105954.png" alt="PCB Inspector Live Camera Inference" width="300" />
  <img src="assets/Screenshot%202026-11-20%20110128.png" alt="PCB Inspector CAM Heatmap Overlay" width="300" />
</p>

## Repository Structure

```text
├── dataset/                # Dataset v1–v4 splits & class statistics
├── labeler/                # Browser-based PCB Component Labeler tool
├── training/               # Teacher fine-tuning, 5-fold CV & KD grid search (T, α)
├── evaluation/             # Ablation studies, confusion matrices & offline Grad-CAM
├── export/                 # TFLite float32, INT8 PTQ & two-output conversion scripts
├── android_app/            # Kotlin Android Studio project (PCB Inspector)
└── assets/                 # Banner, architecture diagrams & app screenshots
```

## Author & Acknowledgements

- **Author:** Nguyen Duc Anh (23BI14013) — ICT Department, University of Science and Technology of Hanoi (USTH)
- **External Supervisor:** Mr. Pham Xuan Long — Foxconn Industrial Internet (Fii)
- **Internal Supervisor:** Mr. Kieu Quoc Viet — University of Science and Technology of Hanoi (USTH)

## License

This project is licensed under the [MIT License](LICENSE).