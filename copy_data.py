#!/usr/bin/env python3
"""
copy_dataset.py — Copy dataset v2 → v4 an toàn (để mở rộng).

Mục đích:
  - Sao chép NGUYÊN cấu trúc dataset_v2 sang dataset_v4
  - GIỮ NGUYÊN v2 gốc (không đụng) → có thể quay lại
  - Sau đó label thêm ảnh vào v4 (pcb_labeler trỏ vào v4)

Cách dùng:
  python3 copy_dataset.py <đường_dẫn_v2> <đường_dẫn_v4>

Ví dụ:
  python3 copy_dataset.py dataset_v2 dataset_v4
  python3 copy_dataset.py "C:/Users/TUF/dataset_v2" "C:/Users/TUF/dataset_v4"

Nếu không truyền tham số → dùng mặc định: dataset_v2 → dataset_v4
"""
import sys
import shutil
from pathlib import Path


def count_images(folder: Path):
    """Đếm số ảnh mỗi lớp trong dataset (train+val+test)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = folder / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = sum(1 for f in cls_dir.iterdir()
                        if f.suffix.lower() in exts)
                counts[cls_dir.name] = counts.get(cls_dir.name, 0) + n
    return counts


def main():
    # đọc tham số
    if len(sys.argv) >= 3:
        src = Path(sys.argv[1])
        dst = Path(sys.argv[2])
    else:
        src = Path("dataset_v2")
        dst = Path("dataset_v4")
        print("⚠️  Không có tham số → dùng mặc định: dataset_v2 → dataset_v4\n")

    print("=" * 55)
    print("COPY DATASET (mở rộng an toàn)")
    print("=" * 55)
    print(f"Nguồn (giữ nguyên): {src.resolve()}")
    print(f"Đích (tạo mới):     {dst.resolve()}")

    # kiểm tra nguồn tồn tại
    if not src.exists():
        print(f"\n❌ KHÔNG tìm thấy dataset nguồn: {src}")
        print("   Kiểm tra lại đường dẫn.")
        return

    # cảnh báo nếu đích đã tồn tại
    if dst.exists():
        print(f"\n⚠️  {dst} ĐÃ TỒN TẠI!")
        ans = input("   Ghi đè? (yes/no): ").strip().lower()
        if ans != "yes":
            print("   Hủy. Không làm gì cả.")
            return
        print("   Xóa đích cũ...")
        shutil.rmtree(dst)

    # đếm ảnh nguồn TRƯỚC khi copy
    print("\n📊 Dataset nguồn (v2):")
    src_counts = count_images(src)
    for cls, n in src_counts.items():
        print(f"   {cls:<12}: {n}")
    print(f"   {'TỔNG':<12}: {sum(src_counts.values())}")

    # COPY
    print(f"\n🔄 Đang copy {src.name} → {dst.name}...")
    shutil.copytree(src, dst)
    print("   ✅ Copy xong")

    # verify: đếm lại ở đích
    print("\n🔍 Kiểm tra dataset đích (v4):")
    dst_counts = count_images(dst)
    for cls, n in dst_counts.items():
        print(f"   {cls:<12}: {n}")
    print(f"   {'TỔNG':<12}: {sum(dst_counts.values())}")

    # so sánh
    if src_counts == dst_counts:
        print("\n✅ THÀNH CÔNG — v4 giống hệt v2")
        print(f"\n📝 BƯỚC TIẾP:")
        print(f"   1. v2 GIỮ NGUYÊN (backup an toàn)")
        print(f"   2. Label thêm ảnh vào {dst.name} (pcb_labeler trỏ vào đây)")
        print(f"   3. Train trên {dst.name}")
    else:
        print("\n⚠️  Số ảnh không khớp — kiểm tra lại!")


if __name__ == "__main__":
    main()