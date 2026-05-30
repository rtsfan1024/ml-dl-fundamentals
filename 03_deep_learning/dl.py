import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# =========================
# 一键配置区：改这里就行
# =========================
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
    EPOCHS: int = 15
    SEED: int = 42
    NUM_WORKERS: int = 2
    DEVICE: str = "auto"  # "cuda" | "mps" | "cpu" | "auto"

    # 早停
    EARLY_STOP: bool = True
    PATIENCE: int = 5  # 连续多少个epoch val_loss无提升则停止

    # 损失函数: cross_entropy | mse (分类建议cross_entropy)
    LOSS: str = "cross_entropy"

    # 数据相关
    NUM_CLASSES: int = 10
    IMG_SIZE: int = 32
    TRAIN_SAMPLES: int = 20000
    VAL_RATIO: float = 0.1
    AUGMENT: bool = True

CFG = Config()

# 固定随机种子
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG.SEED)

# 设备选择
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

# =========================
# 数据：FakeData
# =========================
mean = (0.5, 0.5, 0.5)
std  = (0.5, 0.5, 0.5)
train_tfms = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(CFG.IMG_SIZE, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
] if CFG.AUGMENT else [
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
]
test_tfms = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
]

train_dataset_full = datasets.FakeData(
    size=CFG.TRAIN_SAMPLES,
    image_size=(3, CFG.IMG_SIZE, CFG.IMG_SIZE),
    num_classes=CFG.NUM_CLASSES,
    transform=transforms.Compose(train_tfms),
)
val_size = int(CFG.TRAIN_SAMPLES * CFG.VAL_RATIO)
train_size = CFG.TRAIN_SAMPLES - val_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

test_dataset = datasets.FakeData(
    size=4000,
    image_size=(3, CFG.IMG_SIZE, CFG.IMG_SIZE),
    num_classes=CFG.NUM_CLASSES,
    transform=transforms.Compose(test_tfms),
)

train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

# =========================
# 模型：可切换激活/归一化/Dropout的小型CNN
# =========================
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
        # 对每个样本的 (C,H,W) 做LN
        return nn.LayerNorm([num_channels, H, W])
    if name == "group":
        g = max(1, min(groups, num_channels))
        # 确保可整除
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if name == "rms":
        # 别名：nn.RMSNorm
        try:
            return nn.RMSNorm([num_channels, H, W])
        except AttributeError:
            # 兼容实现：RMSNorm~LayerNorm的无均值版本（近似用LN代替）
            return nn.LayerNorm([num_channels, H, W])
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name}")

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, activation, norm, dropout_p, H, W):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm = make_norm2d(norm, out_c, H, W)
        self.act  = make_activation(activation)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        C, H, W = 3, cfg.IMG_SIZE, cfg.IMG_SIZE
        self.block1 = ConvBlock(C,   64, cfg.ACTIVATION, cfg.NORM, cfg.DROPOUT_P, H, W)    # -> 64xHxW
        self.pool1  = nn.MaxPool2d(2)                                                       # -> 64xH/2xW/2
        self.block2 = ConvBlock(64, 128, cfg.ACTIVATION, cfg.NORM, cfg.DROPOUT_P, H//2, W//2)
        self.pool2  = nn.MaxPool2d(2)                                                       # -> 128xH/4xW/4
        self.block3 = ConvBlock(128, 256, cfg.ACTIVATION, cfg.NORM, cfg.DROPOUT_P, H//4, W//4)
        self.pool3  = nn.AdaptiveAvgPool2d((1,1))                                           # -> 256x1x1
        self.head   = nn.Linear(256, cfg.NUM_CLASSES)

    def forward(self, x):
        x = self.block1(x); x = self.pool1(x)
        x = self.block2(x); x = self.pool2(x)
        x = self.block3(x); x = self.pool3(x)
        x = torch.flatten(x, 1)
        return self.head(x)

# =========================
# 损失 / 优化器 / 调度器
# =========================
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
        # 线性warmup到1.0，然后余弦衰减（以step计数）
        total_steps = CFG.EPOCHS * steps_per_epoch
        warmup = CFG.COSINE_WARMUP_STEPS

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {name}")

# =========================
# 训练/验证/测试循环+早停
# =========================
def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, scheduler=None, global_step=0):
    model.train()
    total_loss, total_acc, count = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        if isinstance(criterion, nn.MSELoss):
            y_onehot = F.one_hot(yb, num_classes=CFG.NUM_CLASSES).float()
            loss = criterion(logits, y_onehot)
        else:
            loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            # warmup/cosine按step更新
            scheduler.step()
        total_loss += loss.item() * xb.size(0)
        total_acc  += accuracy_from_logits(logits, yb) * xb.size(0)
        count      += xb.size(0)
        global_step += 1
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        # 非step型（按epoch）更新
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass  # 在验证后调用
        else:
            scheduler.step()
    return total_loss / count, total_acc / count, global_step

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
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
    model.eval()
    correct, total = 0, 0
    num_classes = CFG.NUM_CLASSES
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        for t, p in zip(yb.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
        correct += (preds == yb).sum().item()
        total   += yb.numel()
    acc = correct / total
    return acc, cm.cpu()

# =========================
# 主过程
# =========================
def main():
    model = SmallCNN(CFG).to(device)
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

        # ReduceLROnPlateau需要在val后step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{CFG.EPOCHS} | lr={lr_now:.2e} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | {dt:.1f}s")

        # 早停
        if CFG.EARLY_STOP:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= CFG.PATIENCE:
                    print(f"[EarlyStop] patience={CFG.PATIENCE} reached. Stop at epoch {epoch}.")
                    break

    # 恢复最优
    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, cm = test_report(model, test_loader)
    print(f"[TEST] accuracy = {test_acc:.3f}")
    print("[CONFUSION MATRIX]")
    print(cm)

if __name__ == "__main__":
    main()



