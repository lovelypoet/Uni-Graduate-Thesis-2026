#!/usr/bin/env python3
"""
dataset_size_check.py — Đo dung lượng dataset: mỗi ảnh nặng bao nhiêu,
tổng dataset nặng bao nhiêu. Chạy ngay trong thư mục chứa dataset_v4/.

Cách dùng: python3 dataset_size_check.py dataset_v4
"""
import sys
from pathlib import Path

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_v4")
    if not root.exists():
        print(f"❌ Không tìm thấy: {root}")
        return

    sizes = []
    per_class = {}
    for f in root.rglob("*.jpg"):
        sz = f.stat().st_size
        sizes.append(sz)
        cls = f.parent.name
        per_class.setdefault(cls, []).append(sz)

    if not sizes:
        print("❌ Không tìm thấy ảnh .jpg nào")
        return

    total_bytes = sum(sizes)
    avg_bytes = total_bytes / len(sizes)

    print("=" * 50)
    print(f"DATASET SIZE REPORT — {root}")
    print("=" * 50)
    print(f"Tổng số ảnh:        {len(sizes)}")
    print(f"Tổng dung lượng:    {total_bytes/1e6:.2f} MB")
    print(f"Trung bình/ảnh:     {avg_bytes/1024:.1f} KB")
    print(f"Nhỏ nhất:           {min(sizes)/1024:.1f} KB")
    print(f"Lớn nhất:           {max(sizes)/1024:.1f} KB")
    print()
    print("Theo từng class:")
    for cls, szs in sorted(per_class.items()):
        print(f"  {cls:<12}: {len(szs):>4} ảnh, "
              f"trung bình {sum(szs)/len(szs)/1024:.1f} KB, "
              f"tổng {sum(szs)/1e6:.2f} MB")

if __name__ == "__main__":
    main()