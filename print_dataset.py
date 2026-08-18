from pathlib import Path

for ds in ["dataset", "dataset_v2", "dataset_v3", "dataset_v4"]:
    print("="*40)
    print(ds)

    total = 0
    root = Path(ds)

    for cls in sorted((root/"train").iterdir()):
        if cls.is_dir():
            n = len(list(root.glob(f"**/{cls.name}/*.jpg")))
            print(f"{cls.name:12}: {n}")
            total += n

    print("Total:", total)