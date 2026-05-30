# ML & DL Fundamentals

涵盖机器学习、深度学习、自然语言处理、强化学习四大方向的基础实验。每个实验都附有代码实现与可视化输出，记录从原理到落地的完整过程。

---

## 实验环境

| 项目 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.8 |
| CUDA | 13 |

**依赖安装：**

```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas pillow
pip install datasets rouge-score rake-nltk nltk jieba
```

> 深度学习模块推荐 CUDA 环境，CPU 模式可运行但训练较慢。

---

## 目录结构

```
.
├── 01_activation_functions/     # 激活函数实验
│   ├── sigmoid.py
│   ├── softmax.py
│   ├── relu.py
│   ├── tanh.py
│   └── step_func.py
│
├── 02_machine_learning/         # 机器学习实验
│   ├── ml.py                    # 线性回归预测
│   ├── pca.py                   # PCA 降维
│   └── tfidf.py                 # TF-IDF 关键词提取
│
├── 03_deep_learning/            # 深度学习实验
│   ├── img_classifying.py       # 简易 CNN 图像分类
│   ├── dl.py                    # CNN 对比实验 (FakeData)
│   ├── dl2cifar10.py            # CNN 训练 (CIFAR-10)
│   ├── dl2cifar10_ResNet.py     # ResNet18 从零训练
│   ├── dl2cifar10_ResNet_TL.py  # ResNet18 迁移学习
│   └── cifar_data_transfor.py   # CIFAR-10 数据转换工具
│
├── 04_nlp/                      # 自然语言处理实验
│   ├── text_sum.py              # 抽取式文本摘要
│   ├── download_cnn_dailymail.py
│   └── load_dataset2csv.py
│
├── 05_reinforcement_learning/   # 强化学习实验
│   └── rl.py                    # Q-Learning 红绿灯
│
└── assets/                      # 静态资源
    ├── images/                  # 测试图片与演示素材
    │   ├── bat.jpg              # CNN 图像分类测试图片
    │   ├── hourse.png           # 示例图片
    │   └── traffic_light_demo.gif  # 红绿灯 Q-Learning 演示动图
    │
    ├── data/                    # 数据集文件
    │   ├── ecommerce_sales_dataset.csv      # 电商销售数据 (ml.py 使用)
    │   ├── cnn_dailymail_full.csv           # CNN/DailyMail 新闻摘要数据 (text_sum.py 使用)
    │   └── predictions_linear_regression.csv  # 线性回归预测输出结果
    │
    └── models/                  # 模型权重与训练缓存
        ├── data/                # CIFAR-10 数据集 (PyTorch 自动下载格式)
        │   ├── cifar-10-python.tar.gz       # 原始压缩包
        │   ├── cifar-10-batches-py/         # 解压后的批次文件
        │   └── cifar10_min_cnn/             # img_classifying.py 训练产出
        │       └── best_cifar10_cnn.pth     # SmallCNN 最佳权重
        ├── .data/               # 迁移学习产出
        │   ├── cifar-10-python.tar.gz
        │   └── resnet18_cifar10_best.pth    # ResNet18 预训练权重
        └── quick_ckpt/          # NLP 模型检查点
            └── artifacts.pkl    # text_sum.py 训练的 TF-IDF + 分类器
```

---

## 01 - 激活函数实验

> **实验目的：** 理解神经网络中激活函数的作用——引入非线性变换，使网络能够拟合复杂函数。通过可视化直观感受不同激活函数的形状、值域和梯度特性，为后续深度学习打下数学基础。

| 文件 | 函数 | 公式 | 特性 |
|------|------|------|------|
| `sigmoid.py` | Sigmoid | f(x) = 1 / (1 + e^(-x)) | 输出 (0,1)，适合二分类，存在梯度消失问题 |
| `softmax.py` | Softmax | y_k = e^(x_k) / sum(e^(x_i)) | 多分类输出概率分布，保证所有输出之和为 1 |
| `relu.py` | ReLU | f(x) = max(0, x) | 计算简单，缓解梯度消失，是 CNN 默认激活函数 |
| `tanh.py` | Tanh | f(x) = (e^x - e^(-x)) / (e^x + e^(-x)) | 输出 (-1,1)，零中心化，收敛快于 Sigmoid |
| `step_func.py` | 阶跃函数 | f(x) = 1 if x > 0 else 0 | 感知机的基础，不可微，现代网络已不使用 |

**运行方式：**
```bash
cd 01_activation_functions
python sigmoid.py    # 可视化 Sigmoid 曲线
python softmax.py    # 计算 Softmax 概率分布
```

---

## 02 - 机器学习实验

> **实验目的：** 掌握机器学习的核心工作流程——数据预处理、特征工程、模型训练、评估与预测。理解 sklearn Pipeline 如何防止数据泄漏，以及正则化（Ridge/Lasso）如何控制过拟合。同时学习 PCA 降维和 TF-IDF 文本特征提取等经典方法。

### 实验 2.1：线性回归预测 (`ml.py`)

基于电商销售数据集，构建端到端的回归预测 Pipeline。

| 环节 | 实现 |
|------|------|
| 数据预处理 | 缺失值填充（中位数/众数）、One-Hot 编码、标准化 |
| 模型选择 | 线性回归 / Ridge (L2) / Lasso (L1) 可切换 |
| 评估方式 | 固定切分 (MAE/RMSE) 或 K 折交叉验证 |
| 防泄漏设计 | sklearn Pipeline 封装全流程 |

### 实验 2.2：PCA 降维 (`pca.py`)

将 3D 数据降至 2D，理解主成分分析的数学原理：中心化 → 协方差矩阵 → 特征值分解 → 投影。

### 实验 2.3：TF-IDF 关键词提取 (`tfidf.py`)

对中文文本进行分词（jieba），计算词频 (TF) 和逆文档频率 (IDF)，提取权重最高的关键词。

**运行方式：**
```bash
cd 02_machine_learning
python ml.py         # 运行线性回归预测
python pca.py        # 查看 PCA 降维结果
python tfidf.py      # 提取中文关键词
```

---

## 03 - 深度学习实验（图像分类）

> **实验目的：** 从零搭建 CNN 到使用预训练 ResNet，逐步理解卷积神经网络的结构设计、训练技巧与优化策略。掌握数据增强、学习率调度、Early Stopping 等工程化训练方法，并体会迁移学习在小数据场景下的威力。

所有实验基于 **CIFAR-10** 数据集（10 类 32x32 彩色图片，5 万训练 + 1 万测试）。

| 实验 | 文件 | 模型 | 核心要点 |
|------|------|------|----------|
| 3.1 | `img_classifying.py` | SmallCNN | 最简 CNN，支持自定义图片预测，入门首选 |
| 3.2 | `dl.py` | SmallCNN + FakeData | 可切换激活函数/归一化/优化器/调度器，用于对比实验 |
| 3.3 | `dl2cifar10.py` | SmallCNN + CIFAR-10 | 真实数据训练，含数据增强（翻转/裁剪） |
| 3.4 | `dl2cifar10_ResNet.py` | ResNet18 (无预训练) | 从零训练残差网络，体验残差连接的收敛优势 |
| 3.5 | `dl2cifar10_ResNet_TL.py` | ResNet18 (ImageNet 预训练) | 迁移学习，冻结特征层微调分类头，效果最佳 |

**通用训练特性：** AdamW 优化器、Cosine Warmup 学习率调度、Early Stopping、混淆矩阵评估。

**训练难度递进：**
```
FakeData 演示 → 自建 CNN → 残差网络 → 迁移学习
（理解结构）  （真实数据）  （更深网络）  （站在巨人肩膀）
```

**运行方式：**
```bash
cd 03_deep_learning
python img_classifying.py       # 最简图像分类 + 自定义图片预测
python dl2cifar10.py            # CNN 训练
python dl2cifar10_ResNet.py     # ResNet 从零训练
python dl2cifar10_ResNet_TL.py  # 迁移学习（推荐）
```

---

## 04 - 自然语言处理实验

> **实验目的：** 学习 NLP 的基础流程——从数据获取、文本预处理到特征提取与建模。通过抽取式文本摘要任务，理解 TF-IDF 向量化、句-文相似度计算、以及如何用分类模型判断句子是否应该被选入摘要。

### 实验 4.1：数据准备

| 文件 | 功能 |
|------|------|
| `load_dataset2csv.py` | 从 HuggingFace 下载 CNN/DailyMail 数据集并转为 CSV |
| `download_cnn_dailymail.py` | 另一种下载方式（逐条写入，支持断点续传） |

### 实验 4.2：抽取式文本摘要 (`text_sum.py`)

基于 CNN/DailyMail 新闻数据集，实现抽取式摘要 + 关键词提取。

| 环节 | 实现 |
|------|------|
| 句子切分 | NLTK sent_tokenize（正则兜底） |
| 特征工程 | TF-IDF 句向量、句-文余弦相似度、位置特征 |
| 句子打分 | Logistic Regression 二分类（ROUGE-L 弱监督标注） |
| 多样性去重 | 候选句之间余弦相似度 > 0.6 则跳过 |
| 关键词提取 | TF-IDF 权重排序 + RAKE 兜底 |

**运行方式：**
```bash
cd 04_nlp
python load_dataset2csv.py                    # 下载数据集
python text_sum.py --mode train               # 训练模型
python text_sum.py --mode predict --text "..." # 摘要推理
```

---

## 05 - 强化学习实验

> **实验目的：** 理解强化学习的基本框架——智能体 (Agent)、环境 (Environment)、状态 (State)、动作 (Action)、奖励 (Reward)。通过红绿灯场景，掌握 Q-Learning 算法的核心：贝尔曼方程更新、Epsilon-Greedy 探索与利用平衡、折扣因子对远视/短视的影响。

### 实验 5.1：Q-Learning 红绿灯决策 (`rl.py`)

模拟一辆车在直路上行驶，遇到红绿灯时需要决定"前进"还是"等待"。

| 概念 | 对应实现 |
|------|----------|
| 状态空间 | (汽车位置, 红灯/绿灯) 共 18 种组合 |
| 动作空间 | STOP (等待) / GO (前进) |
| 奖励设计 | 闯红灯 -10，停车等待 -0.1，到达终点 +5 |
| 算法 | Q-Learning + Epsilon-Greedy (eps 从 1.0 衰减到 0.05) |
| 可视化 | matplotlib 动画，支持键盘交互（暂停/单步/切换AI/随机） |

**运行方式：**
```bash
cd 05_reinforcement_learning
python rl.py    # 训练 320 局后自动启动动画可视化
```

---

## 环境依赖

```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas pillow
pip install datasets rouge-score rake-nltk nltk jieba
```

> 深度学习模块推荐 CUDA 环境，CPU 模式可运行但训练较慢。
