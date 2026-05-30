from pathlib import Path
import csv
from datasets import load_dataset
from tqdm import tqdm
import gzip

# ======== 配置（按需修改）========
SAVE_PATH = Path("/Users/zhaoshuai/Downloads/cnn_dailymail_full.csv")  # 也可改成 .csv.gz
SPLITS = ["train", "validation", "test"]  # 需要哪些分割
CONFIG_NAME = "3.0.0"
# =================================

def open_outfile(path: Path):
    """根据后缀返回文件句柄和写入器（支持 .csv 和 .csv.gz）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        f = gzip.open(path, "wt", newline="", encoding="utf-8")
    else:
        f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    return f, writer

def main():
    # 提示：首次运行会从HF下载分片文件，datasets会自带下载进度
    print(f"[INFO] 目标文件: {SAVE_PATH}")
    print(f"[INFO] 数据配置: cnn_dailymail / {CONFIG_NAME} / {SPLITS}")

    fout, writer = open_outfile(SAVE_PATH)
    # 统一表头
    header = ["id", "article", "highlights", "split"]
    writer.writerow(header)

    total_rows = 0
    try:
        for split in SPLITS:
            print(f"\n[STEP] 加载 split = {split} ...")
            ds = load_dataset("cnn_dailymail", CONFIG_NAME, split=split)
            n = len(ds)
            print(f"[INFO] {split}: {n} 条")

            # 逐条写入（带进度条）
            for ex in tqdm(ds, total=n, desc=f"Writing {split}", unit="row"):
                # 保证字段存在（极端情况下做个兜底）
                _id = ex.get("id", "")
                article = ex.get("article", "")
                highlights = ex.get("highlights", "")
                writer.writerow([_id, article, highlights, split])
                total_rows += 1
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断，已写入的内容保留。")
    finally:
        fout.flush()
        fout.close()

    print(f"\n[DONE] 写入完成：{SAVE_PATH}")
    print(f"[STATS] 总行数：{total_rows}（含 train/validation/test 合并）")

if __name__ == "__main__":
    # 可选：如果需要代理，运行脚本前在shell设置
    # export HTTP_PROXY=http://127.0.0.1:7890
    # export HTTPS_PROXY=http://127.0.0.1:7890
    main()



