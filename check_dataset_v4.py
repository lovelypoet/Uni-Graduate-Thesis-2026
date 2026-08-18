#!/usr/bin/env python3
"""
check_dataset_v4.py — Kiểm tra nhanh distribution của dataset_v4
trước khi chạy lại toàn bộ pipeline (CV, training, KD sweep).

Chạy: python3 check_dataset_v4.py
"""
import config as C
from pathlib import Path

print("=" * 60)
print(f"DATASET CHECK: {C.DATASET_DIR}")
print("=" * 60)

total_all = 0
for split in ["train", "val", "test"]:
    split_dir = C.DATASET_DIR / split
    if not split_dir.exists():
        print(f"\n⚠️  {split}/ không tồn tại!")
        continue

    print(f"\n{split.upper()}:")
    split_total = 0
    for cls in C.CLASS_NAMES:
        cls_dir = split_dir / cls
        n = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0
        split_total += n
        bar = "█" * (n // 5)
        print(f"  {cls:12s}: {n:4d}  {bar}")
    print(f"  {'TOTAL':12s}: {split_total:4d}")
    total_all += split_total

print(f"\n{'='*60}")
print(f"GRAND TOTAL: {total_all} images")
print(f"{'='*60}")

# Imbalance ratio
weights, counts = None, {}
train_dir = C.DATASET_DIR / "train"
for i, cls in enumerate(C.CLASS_NAMES):
    cls_dir = train_dir / cls
    counts[cls] = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0

if counts:
    max_c = max(counts.values())
    min_c = min(counts.values())
    print(f"\nImbalance ratio (train): {max_c}/{min_c} = {max_c/max(min_c,1):.1f}x")
    print(f"Largest class : {max(counts, key=counts.get)} ({max_c})")
    print(f"Smallest class: {min(counts, key=counts.get)} ({min_c})")

print("\n→ Nếu distribution OK, chạy tiếp:")
print("   python3 cross_validation.py     # re-validate teacher trên data mới")