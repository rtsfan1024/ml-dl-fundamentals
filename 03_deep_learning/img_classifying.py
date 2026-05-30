from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# =============== 可配参数 =================
BASE_DIR       = Path(__file__).resolve().parent.parent             # 项目根目录
DATA_DIR       = BASE_DIR / "assets" / "models" / "data"           # CIFAR-10 缓存/下载目录
SAVE_DIR       = DATA_DIR / "cifar10_min_cnn"                      # 模型与日志保存目录
NUM_EPOCHS     = 5                                  # 训练轮数（示例用小轮数数）
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
MOMENTUM       = 0.9                              # 用在SGD时；如果用Adam可忽略
OPTIMIZER      = "adam"                           # "adam"或"sgd"
USE_AUG        = True                             # 是否启用数据增强（随机裁剪/翻转）
NUM_WORKERS    = 2
SEED           = 42
CKPT_NAME = "best_cifar10_cnn.pth"   # 统一管理权重文件名
SKIP_TRAIN_IF_CKPT_EXISTS = True     # 有权重且提供IMG_PATH时，直接跳过训练去预测

# 指定一张要预测的图片（任意来源的RGB图片），空字符串则跳过预测
IMG_PATH       = str(BASE_DIR / "assets" / "images" / "bat.jpg")   # 例如："/Users/zhaoshuai/Downloads/zebra.jpg"
# ========================================

# 固定随机种子（结果更可复现）
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 简单的CNN（32x32输入）
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32x32
        self.pool  = nn.MaxPool2d(2, 2)                           # -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 16x16
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)# 16x16
        self.pool2 = nn.MaxPool2d(2, 2)                           # -> 8x8
        self.dropout = nn.Dropout(0.3)
        self.fc1   = nn.Linear(128*8*8, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_transforms(train=True):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    if train and USE_AUG:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

def prepare_data(device):
    assert (DATA_DIR / "cifar-10-batches-py").exists(), \
        f"找不到本地数据目录：{DATA_DIR/'cifar-10-batches-py'}"

    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=False, transform=get_transforms(train=True)
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=False, transform=get_transforms(train=False)
    )

    pin = (device.type == "cuda")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    classes = train_set.classes
    return train_loader, test_loader, classes

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc  += (logits.argmax(1) == labels).float().sum().item()
        n += bs
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc  += (logits.argmax(1) == labels).float().sum().item()
        n += bs
    return running_loss / n, running_acc / n

def get_optimizer(model):
    if OPTIMIZER.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                               momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)

def predict_image(model, img_path, classes, device):
    """
    对任意图片做分类：会自动RGB+resize(32,32)+normalize
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()
    topk = probs.argsort()[::-1][:3]
    print(f"\n[Predict] {img_path}")
    for i, k in enumerate(topk, 1):
        print(f"Top-{i}: {classes[k]}  prob={probs[k]:.4f}")
    return classes[topk[0]]

def main():
    ckpt_path = SAVE_DIR / CKPT_NAME

    if SKIP_TRAIN_IF_CKPT_EXISTS and not IMG_PATH and ckpt_path.exists():
        print(f"[Skip] 已存在训练好的模型：{ckpt_path}，未指定图片，因此直接退出。")
        return

    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    print(f"[Device] {device}")


    # 如果：配置允许跳过训练+提供了图片路径+ckpt已存在->直接加载预测并 return
    if SKIP_TRAIN_IF_CKPT_EXISTS and IMG_PATH and Path(IMG_PATH).exists() and ckpt_path.exists():
        print(f"[FastPredict] 发现已训练权重：{ckpt_path}，直接加载进行预测。")
        ckpt = torch.load(ckpt_path, map_location=device)
        classes = ckpt["classes"]
        model = SmallCNN(num_classes=len(classes)).to(device)
        model.load_state_dict(ckpt["model_state"])
        predict_image(model, IMG_PATH, classes, device)
        return

    train_loader, test_loader, classes = prepare_data(device)
    model = SmallCNN(num_classes=len(classes)).to(device)
    optimizer = get_optimizer(model)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"test_loss={te_loss:.4f} acc={te_acc:.4f}")

        # 保存最好模型
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "config": {
                    "lr": LEARNING_RATE, "wd": WEIGHT_DECAY, "optimizer": OPTIMIZER,
                    "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE
                }
            }, SAVE_DIR / CKPT_NAME)
            print(f"[Save] 更新最佳模型 acc={best_acc:.4f} -> {SAVE_DIR / CKPT_NAME}")

    # 可选：对任意图片做一次预测
    if IMG_PATH and Path(IMG_PATH).exists():
        ckpt = torch.load(SAVE_DIR / CKPT_NAME, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        predict_image(model, IMG_PATH, classes, device)
    else:
        if IMG_PATH:
            print(f"[Warn] 指定的图片不存在：{IMG_PATH}")

if __name__ == "__main__":
    main()

