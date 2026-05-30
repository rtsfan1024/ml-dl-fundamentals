import csv
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image

# ==== 配置（按需修改）====
_BASE_DIR  = Path(__file__).resolve().parent.parent          # 项目根目录
DATA_DIR   = _BASE_DIR / "assets" / "models" / "data" / "cifar-10-batches-py"   # 你的输入目录
OUTPUT_DIR = _BASE_DIR / "assets" / "models" / "data" / "cifar10_images"        # 输出图片目录
NUM_PER_CLASS = None  # 每类最多导出多少张；None 表示全部
# ========================

def unpickle(file: str) -> Dict:
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")

def ensure_dirs(class_names: List[str], out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    for c in class_names:
        (out_root / c).mkdir(parents=True, exist_ok=True)

def export_split(batch_file: str) -> str:
    return "test" if "test_batch" in batch_file else "train"

def save_image(img_flat: np.ndarray, save_path: Path):
    # CIFAR-10: (3072,) -> (3,32,32) -> (32,32,3) uint8
    img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
    Image.fromarray(img).save(save_path)

def main():
    data_dir = Path(DATA_DIR)
    out_dir  = Path(OUTPUT_DIR)
    assert data_dir.exists(), f"输入目录不存在：{data_dir}"

    # 读取类别名
    meta = unpickle(str(data_dir / "batches.meta"))
    class_names = [c.decode("utf-8") for c in meta[b"label_names"]]

    # 创建输出目录
    ensure_dirs(class_names, out_dir)

    # 要处理的批次文件
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]

    # 每类计数器
    counters = {c: 0 for c in class_names}

    # manifest.csv
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filepath", "label", "label_id", "split"])

        for bf in batch_files:
            bf_path = data_dir / bf
            if not bf_path.exists():
                print(f"[WARN] 跳过不存在的批次文件：{bf_path}")
                continue

            print(f"[INFO] 处理 {bf_path.name} ...")
            d = unpickle(str(bf_path))
            data = d[b"data"]            # shape: (10000, 3072)
            labels = d[b"labels"]        # list[int]
            split = export_split(bf)

            for i, img_flat in enumerate(data):
                label_id = labels[i]
                cls_name = class_names[label_id]

                # 数量限制
                if NUM_PER_CLASS is not None and counters[cls_name] >= NUM_PER_CLASS:
                    continue

                # 保存路径
                filename = f"{cls_name}_{counters[cls_name]:05d}.png"
                save_path = out_dir / cls_name / filename

                save_image(img_flat, save_path)
                counters[cls_name] += 1

                writer.writerow([str(save_path), cls_name, label_id, split])

    print("\n=== 转换完成 ✅ ===")
    for c in class_names:
        print(f"{c:>10s}: {counters[c]} 张 -> {out_dir / c}")
    print(f"清单文件：{manifest_path}")

if __name__ == "__main__":
    main()



