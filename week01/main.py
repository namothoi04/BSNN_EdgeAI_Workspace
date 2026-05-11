"""
main.py

CNN baseline tối giản cho MNIST (PyTorch)
- Kiến trúc gọn: Conv -> ReLU -> MaxPool -> Flatten -> Linear
- Mục tiêu: dễ đọc, dễ sửa, dễ dùng làm baseline để chuyển sang BNN / SNN / BSNN
- Phù hợp cho người mới nhập môn

Cách chạy:
    python main.py

Tùy chọn:
    python main.py --epochs 10 --batch_size 32 --lr 0.001
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. Hàm cố định seed
# ------------------------------------------------------------
# Giúp kết quả ổn định hơn giữa các lần chạy
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Đặt deterministic=True để ưu tiên tính tái lập
    # benchmark=False để tránh thay đổi thuật toán nội bộ giữa các lần chạy
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Chọn thiết bị chạy
# ------------------------------------------------------------
# Nếu có GPU thì dùng GPU, ngược lại dùng CPU
# ============================================================
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3. Định nghĩa mô hình CNN baseline
# ------------------------------------------------------------
# Kiến trúc:
#   Input: 1x28x28
#   Conv2d(1 -> 32, 3x3)
#   ReLU
#   MaxPool2d(2x2)
#   Flatten
#   Linear(32*13*13 -> 10)
#
# Đây là bản rất gọn để:
# - dễ hiểu
# - dễ chuyển sang BNN (thay Conv/Linear thành lớp nhị phân)
# - dễ chuyển sang SNN (giữ backbone, thay cơ chế neuron)
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Khối trích xuất đặc trưng
        self.conv1 = nn.Conv2d(
            in_channels=1,   # ảnh MNIST là ảnh xám: 1 kênh
            out_channels=32,
            kernel_size=3
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Khối phân loại
        # Sau Conv3x3 từ 28x28 -> 26x26
        # Sau MaxPool2x2 -> 13x13
        # Số đặc trưng = 32 * 13 * 13
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 13 * 13, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)     # [B, 1, 28, 28] -> [B, 32, 26, 26]
        x = self.relu(x)
        x = self.pool(x)      # [B, 32, 26, 26] -> [B, 32, 13, 13]
        x = self.flatten(x)   # [B, 32, 13, 13] -> [B, 5408]
        x = self.fc(x)        # [B, 5408] -> [B, 10]
        return x              # trả về logits (chưa softmax)


# ============================================================
# 4. Tạo DataLoader
# ------------------------------------------------------------
# Chia train thành train/validation
# Test giữ riêng để đánh giá cuối cùng
# ============================================================
def build_dataloaders(batch_size: int, val_ratio: float, seed: int):
    transform = transforms.ToTensor()   # chuẩn hóa pixel về [0, 1]

    full_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# 5. Hàm train 1 epoch
# ------------------------------------------------------------
# Chức năng:
# - đặt mô hình vào chế độ train: model.train()
# - duyệt toàn bộ các mini-batch của tập train
# - với mỗi batch:
#     1) đưa dữ liệu lên device
#     2) xóa gradient cũ
#     3) forward để lấy logits
#     4) tính loss
#     5) backpropagation: loss.backward()
#     6) cập nhật trọng số: optimizer.step()
# - cộng dồn loss và accuracy của toàn bộ epoch
#
# Kết quả trả về:
# - avg_loss: loss trung bình trên toàn bộ tập train của epoch đó
# - avg_acc : accuracy trung bình trên toàn bộ tập train của epoch đó
#
# Đây là hàm quan trọng nhất để sinh viên hiểu "mô hình học như thế nào".
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


# ============================================================
# 6. Hàm đánh giá
# ------------------------------------------------------------
# Chức năng:
# - dùng cho validation hoặc test
# - đặt mô hình sang chế độ đánh giá: model.eval()
# - tắt gradient bằng @torch.no_grad() để:
#     + tiết kiệm bộ nhớ
#     + tăng tốc
#     + tránh cập nhật trọng số ngoài ý muốn
# - chỉ forward để đo loss và accuracy, KHÔNG backpropagation
#
# Khác với train_one_epoch():
# - train_one_epoch() có học (có backward + optimizer.step)
# - evaluate() chỉ đo chất lượng mô hình trên dữ liệu chưa dùng để cập nhật
#
# Kết quả trả về:
# - avg_loss: loss trung bình trên tập validation hoặc test
# - avg_acc : accuracy trung bình trên tập validation hoặc test
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


# ============================================================
# 7. Đếm số tham số
# ============================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================
# 8. Vẽ đồ thị huấn luyện
# ============================================================
def plot_history(history, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    ax1.plot(epochs, history["val_acc"], marker="x", label="Validation Accuracy")
    best_epoch_acc = int(np.argmax(history["val_acc"])) + 1
    best_acc = float(np.max(history["val_acc"]))
    ax1.axvline(x=best_epoch_acc, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(f"Best: {best_acc*100:.2f}%", xy=(best_epoch_acc, best_acc),
                 xytext=(best_epoch_acc - 1.5, best_acc - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Loss
    ax2.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax2.plot(epochs, history["val_loss"], marker="x", label="Validation Loss")
    best_epoch_loss = int(np.argmin(history["val_loss"])) + 1
    best_loss = float(np.min(history["val_loss"]))
    ax2.axvline(x=best_epoch_loss, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f"Best: {best_loss:.4f}", xy=(best_epoch_loss, best_loss),
                 xytext=(best_epoch_loss - 1.5, best_loss + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "history.png", dpi=150)
    plt.savefig(save_dir / "history.svg")
    plt.show()
    plt.close()


# ============================================================
# 9. Chương trình chính
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CNN baseline tối giản cho MNIST")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỉ lệ validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập")
    parser.add_argument("--out_dir", type=str, default="runs_mnist_cnn_clean", help="Thư mục lưu kết quả")
    args = parser.parse_args()

    # Khởi tạo
    seed_everything(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MNIST CNN BASELINE (CLEAN VERSION)")
    print(f"Device      : {device}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Seed        : {args.seed}")
    print("=" * 60)

    # Dữ liệu
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Mô hình
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    print(f"Total parameters: {count_parameters(model):,}")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    # Huấn luyện
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Lưu checkpoint tốt nhất theo validation
    if best_state is not None:
        torch.save(best_state, out_dir / "best_model.pt")
        model.load_state_dict(best_state)

    # Đánh giá cuối trên test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("-" * 60)
    print(f"Best validation accuracy : {best_val_acc*100:.2f}% at epoch {best_epoch}")
    print(f"Test loss               : {test_loss:.4f}")
    print(f"Test accuracy           : {test_acc*100:.2f}%")
    print(f"Saved checkpoint        : {out_dir / 'best_model.pt'}")
    print("-" * 60)

    # Vẽ đồ thị
    plot_history(history, out_dir)

    # Lưu lịch sử để tiện xem lại / chuyển sang BNN/SNN sau này
    np.save(out_dir / "train_loss.npy", np.array(history["train_loss"]))
    np.save(out_dir / "train_acc.npy", np.array(history["train_acc"]))
    np.save(out_dir / "val_loss.npy", np.array(history["val_loss"]))
    np.save(out_dir / "val_acc.npy", np.array(history["val_acc"]))


if __name__ == "__main__":
    main()
