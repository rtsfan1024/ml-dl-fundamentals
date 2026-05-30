import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录

# 1. 强制使用国内加速镜像，解决下载网络报错问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("1. 正在连接并下载 cnn_dailymail 数据集 (压缩包约 800MB，请耐心等待)...")
# 强制指定最常用的非匿名化 3.0.0 版本
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

print("2. 数据下载完成！正在将拆分的数据转换为表格并合并...")
# 将 训练集、验证集、测试集 全部提取并转为 Pandas DataFrame
df_train = dataset["train"].to_pandas()
df_val = dataset["validation"].to_pandas()
df_test = dataset["test"].to_pandas()

# 将三份试卷纵向无缝拼接成一个完整的超大表格
df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)

print(f"3. 合并成功！总计包含 {len(df_full)} 篇新闻文章。正在写入 CSV 文件...")
# 一键导出为 csv，index=False 代表不保留无用的行号索引
df_full.to_csv(str(BASE_DIR / "assets" / "data" / "cnn_dailymail_full.csv"), index=False)

print("4. 🎉 大功告成！【cnn_dailymail_full.csv】 已经成功生成在当前目录！")