#!/usr/bin/env python3
"""
Đổi tên class 'wrong_part' -> 'flipped' trong dataset.
Đổi cả tên FOLDER lẫn tên FILE (wrong_part_0001.jpg -> flipped_0001.jpg).
Chạy: python3 rename_class.py
"""
from pathlib import Path

DATASET = Path("dataset")
OLD = "wrong_part"
NEW = "flipped"

splits = ["train", "val", "test"]
total_files = 0
total_folders = 0

for split in splits:
    old_dir = DATASET / split / OLD
    new_dir = DATASET / split / NEW
    if not old_dir.exists():
        print(f"  (bỏ qua) {old_dir} không tồn tại")
        continue

    new_dir.mkdir(parents=True, exist_ok=True)
    # đổi tên từng file
    for f in sorted(old_dir.glob("*.jpg")):
        new_name = f.name.replace(OLD, NEW)
        target = new_dir / new_name
        f.rename(target)
        total_files += 1
    # xoá folder cũ (đã rỗng)
    try:
        old_dir.rmdir()
        total_folders += 1
    except OSError:
        print(f"  ⚠ {old_dir} chưa rỗng, kiểm tra lại")

print(f"\n✅ Đổi xong: {total_files} file, {total_folders} folder")
print(f"   {OLD} -> {NEW}")