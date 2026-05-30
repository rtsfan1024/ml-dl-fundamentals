# encoding=utf-8

import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision import models

# =========================================================
# 一键配置区：在这里统管全局超参数
# =========================================================
@dataclass
class Config:
    # 激活函数: relu | gelu | tanh | sigmoid
    ACTIVATION: str = "relu"
    # 归一化层: batch | layer | group | rms | none
    NORM: str = "batch"
    GROUPS: int = 8  # GroupNorm分组数（通道须能被整除）
    DROPOUT_P: float = 0.1

    # 优化器: sgd | momentum | rmsprop | adam | adamw
    OPTIMIZER: str = "adamw"
    LR: float = 3e-4
    WEIGHT_DECAY: float = 0.01

    # 调度器: none | step | cosine | cosine_warmup | reduce_on_plateau
    SCHEDULER: str = "cosine_warmup"
    STEP_DECAY_GAMMA: float = 0.5
    STEP_DECAY_EVERY: int = 5
    COSINE_WARMUP_STEPS: int = 200  # 以step计数
    T_MAX_EPOCHS: int = 20          # CosineAnnealingLR的周期（以epoch计）

    # 训练循环
    BATCH_SIZE: int = 128
    # EPOCHS: int = 15
    EPOCHS: int = 50
    SEED: int = 42
    NUM_WORKERS: int = 2
    DEVICE: str = "auto"  # "cuda" | "mps" | "cpu" | "auto"

    # 早停机制参数
    EARLY_STOP: bool = True
    PATIENCE: int = 5  # 连续多少个epoch val_loss无提升则停止

    # 损失函数: cross_entropy | mse (分类任务强烈建议用 cross_entropy)
    LOSS: str = "cross_entropy"

    # 数据相关 (已按真正的 CIFAR-10 数据集属性修正)
    NUM_CLASSES: int = 10         # 10 个不同的物体类别
    IMG_SIZE: int = 32            # 图像物理分辨率 32x32
    TRAIN_SAMPLES: int = 50000    # CIFAR-10 训练集一共 5 万张图片
    VAL_RATIO: float = 0.1
    AUGMENT: bool = True

CFG = Config()

BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录

# 固定随机种子（为了保证每次跑出来的实验结果可以复现）
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG.SEED)

# 设备全自动选择（优先找显卡）
def pick_device():
    if CFG.DEVICE != "auto":
        return torch.device(CFG.DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print(f"[INFO] device = {device}")

# =========================================================
# 数据加载与处理区：真实 CIFAR10 接入点
# =========================================================
# 这是 CIFAR-10 官方统计出的真实 RGB 三通道均值和标准差
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

# 训练集数据增强：让模型每次看到的图片都有点不一样，防止死记硬背
train_tfms = [
    transforms.RandomHorizontalFlip(),               # 50% 概率左右镜像翻转
    transforms.RandomRotation(15), # 新增：随机旋转15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 新增：颜色和亮度抖动
    transforms.RandomCrop(CFG.IMG_SIZE, padding=4),  # 边缘填充4个像素后再随机裁回32x32
    transforms.ToTensor(),                           # 转为 PyTorch 能看懂的张量格式
    transforms.Normalize(mean, std),                 # 标准化：让数据分布集中在 0 附近
] if CFG.AUGMENT else [
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
]

# 测试集不能搞花里胡哨的增强，必须原汁原味
test_tfms = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
]

# 核心修改：加载 CIFAR10 训练集
# 注意：root="./data" 且 download=True 时，它会去读你放在本地的 cifar-10-python.tar.gz 并自动解压
train_dataset_full = datasets.CIFAR10(
    root=str(BASE_DIR / "assets" / "models" / "data"),
    train=True,
    download=True, 
    transform=transforms.Compose(train_tfms),
)

# 按比例切分出验证集（比如 50000 张里，45000 训练，5000 验证）
val_size = int(CFG.TRAIN_SAMPLES * CFG.VAL_RATIO)
train_size = CFG.TRAIN_SAMPLES - val_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# 核心修改：加载 CIFAR10 测试集（10000 张期末考试卷）
test_dataset = datasets.CIFAR10(
    root=str(BASE_DIR / "assets" / "models" / "data"),
    train=False,
    download=True,
    transform=transforms.Compose(test_tfms),
)

# 打包为 DataLoader，负责在训练时把数据按 Batch 喂给模型
train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

# =========================================================
# 模型定义：SmallCNN (卷积神经网络)
# =========================================================
def make_activation(name: str):
    name = name.lower()
    if name == "relu":    return nn.ReLU(inplace=True)
    if name == "gelu":    return nn.GELU()
    if name == "tanh":    return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")

def make_norm2d(name: str, num_channels: int, H: int, W: int, groups: int = 8):
    name = name.lower()
    if name == "batch":
        return nn.BatchNorm2d(num_channels)
    if name == "layer":
        return nn.LayerNorm([num_channels, H, W])
    if name == "group":
        g = max(1, min(groups, num_channels))
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if name == "rms":
        try:
            return nn.RMSNorm([num_channels, H, W])
        except AttributeError:
            return nn.LayerNorm([num_channels, H, W])
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name}")

class ConvBlock(nn.Module):
    """标准的积木块：卷积特征提取 -> 归一化防梯度消失 -> 激活函数加非线性 -> Dropout防过拟合"""
    def __init__(self, in_c, out_c, activation, norm, dropout_p, H, W):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm = make_norm2d(norm, out_c, H, W)
        self.act  = make_activation(activation)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))

class SmallCNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        C, H, W = 3, cfg.IMG_SIZE, cfg.IMG_SIZE
        # 阶段 1：找基础边缘特征。输出变厚 (64通道)，尺寸减半 (16x16)
        self.block1 = ConvBlock(C,   64, cfg.ACTIVATION, cfg.NORM, cfg.DROPOUT_P, H, W)
        self.pool1  = nn.MaxPool2d(2)                                                      
        # 阶段 2：找轮廓。输出变厚 (128通道)，尺寸减半 (8x8)
        self.block2 = ConvBlock(64, 128, cfg.ACTIVATION, cfg.NORM, cfg.DROPOUT_P, H//2, W//2)
        self.pool2  = nn.MaxPool2d(2)                                                      
        # 阶段 3：找高级特征。输出变厚 (256通道)
        self.block3 = ConvBlock(128, 256, cfg.ACTIVATION, cfg.NORM, cfg.DROPOUT_P, H//4, W//4)
        # 终极池化：把空间上的每个通道强行浓缩成一个点 (256x1x1)
        self.pool3  = nn.AdaptiveAvgPool2d((1,1))                                          
        # 分类器头：256 个特征数字，打分给 10 个类别
        self.head   = nn.Linear(256, cfg.NUM_CLASSES)

    def forward(self, x):
        x = self.block1(x); x = self.pool1(x)
        x = self.block2(x); x = self.pool2(x)
        x = self.block3(x); x = self.pool3(x)
        x = torch.flatten(x, 1)  # 拍平操作，把多维特征变成一维向量，好丢给全连接层
        return self.head(x)

# =========================================================
# 损失 / 优化器 / 调度器
# =========================================================
def make_criterion(name: str):
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {name}")

def make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def make_scheduler(name: str, optimizer, *, steps_per_epoch=None):
    name = name.lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=CFG.STEP_DECAY_EVERY, gamma=CFG.STEP_DECAY_GAMMA)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_MAX_EPOCHS)
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
    if name == "cosine_warmup":
        # 预热策略：让学习率前期先慢慢升到最高，稳定后再像余弦曲线一样降落，防初期梯度爆炸
        total_steps = CFG.EPOCHS * steps_per_epoch
        warmup = CFG.COSINE_WARMUP_STEPS

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {name}")

# =========================================================
# 训练/验证/测试循环+早停
# =========================================================
def accuracy_from_logits(logits, y):
    """从网络输出的一堆概率分数里，挑出最高分的那个当做预测类别，并算准确率"""
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, scheduler=None, global_step=0):
    model.train()  # 开启训练模式 (Dropout 和 BatchNorm 才会生效)
    total_loss, total_acc, count = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # 每次算梯度前，记得清空上一次的历史梯度
        logits = model(xb)                     # 前向传播：模型猜结果
        
        if isinstance(criterion, nn.MSELoss):
            y_onehot = F.one_hot(yb, num_classes=CFG.NUM_CLASSES).float()
            loss = criterion(logits, y_onehot)
        else:
            loss = criterion(logits, yb)       # 算算差了多少 (Loss)
            
        loss.backward()                        # 反向传播：把错误归咎于各个参数
        optimizer.step()                       # 优化器发力：更新权重
        
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()
            
        total_loss += loss.item() * xb.size(0)
        total_acc  += accuracy_from_logits(logits, yb) * xb.size(0)
        count      += xb.size(0)
        global_step += 1
        
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass 
        else:
            scheduler.step()
    return total_loss / count, total_acc / count, global_step

@torch.no_grad() # 声明不计算梯度，不仅省显存，速度还飞快
def evaluate(model, loader, criterion):
    """在验证集上摸底考，看看模型目前学得咋样"""
    model.eval() # 开启评估模式
    total_loss, total_acc, count = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        if isinstance(criterion, nn.MSELoss):
            y_onehot = F.one_hot(yb, num_classes=CFG.NUM_CLASSES).float()
            loss = criterion(logits, y_onehot)
        else:
            loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_acc  += accuracy_from_logits(logits, yb) * xb.size(0)
        count      += xb.size(0)
    return total_loss / count, total_acc / count

@torch.no_grad()
def test_report(model, loader):
    """期末大考，输出最终准确率和混淆矩阵"""
    model.eval()
    correct, total = 0, 0
    num_classes = CFG.NUM_CLASSES
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64) # 创建一个全0的 10x10 空白矩阵
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        
        # 挨个对比真实答案 t 和 预测答案 p，并在矩阵对应格子里加 1
        for t, p in zip(yb.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
        correct += (preds == yb).sum().item()
        total   += yb.numel()
    acc = correct / total
    return acc, cm.cpu()

# =========================================================
# 主过程：串联所有逻辑
# =========================================================
def main():
    # model = SmallCNN(CFG).to(device)
    # 换成这三行：加载没经过预训练的空壳 ResNet18，并把最后的输出改成 10 类
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, CFG.NUM_CLASSES)
    model = model.to(device)
    
    criterion = make_criterion(CFG.LOSS)
    optimizer = make_optimizer(CFG.OPTIMIZER, model.parameters(), CFG.LR, CFG.WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader.dataset) / CFG.BATCH_SIZE)
    scheduler = make_scheduler(CFG.SCHEDULER, optimizer, steps_per_epoch=steps_per_epoch)

    print(f"[CFG] {CFG}")
    print(f"[MODEL] params = {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    global_step = 0

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, global_step = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, global_step)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{CFG.EPOCHS} | lr={lr_now:.2e} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | {dt:.1f}s")

        # 核心策略：早停法 (Early Stopping)
        if CFG.EARLY_STOP:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                # 如果这次成绩突破历史最好记录，就把这套模型参数存下来
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                # 如果连续 5 次考试都没打破记录，直接下课
                if bad_epochs >= CFG.PATIENCE:
                    print(f"[EarlyStop] patience={CFG.PATIENCE} reached. Stop at epoch {epoch}.")
                    break

    # 考完试了，把之前保存的表现最完美的那次模型参数拿回来用
    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, cm = test_report(model, test_loader)
    print(f"[TEST] accuracy = {test_acc:.3f}")
    print("[CONFUSION MATRIX]")
    print(cm)

if __name__ == "__main__":
    main()