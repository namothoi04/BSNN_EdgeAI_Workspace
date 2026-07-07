"""
cifar10_cnn_baseline.py

CNN baseline "sạch" cho CIFAR-10 (PyTorch)
-------------------------------------------------
Mục đích của file này:
- Đây là baseline DUY NHẤT, kiến trúc sẽ được giữ NGUYÊN xuyên suốt
  các bước BNN -> SNN -> BSNN tiếp theo (đúng yêu cầu của thầy: so
  sánh phải công bằng, không được đổi kiến trúc giữa chừng).
- Kiến trúc bám sát ĐÚNG baseline MNIST thầy đã cho (Conv -> ReLU
  -> MaxPool -> Flatten -> Linear), chỉ đổi input từ 1 kênh (ảnh
  xám 28x28) sang 3 kênh (ảnh RGB 32x32) và tính lại kích thước
  Flatten cho phù hợp:
    Input (32x32x3)
    -> Conv(3->32, 3x3, không padding) -> 30x30x32
    -> ReLU -> MaxPool(2x2) -> 15x15x32
    -> Flatten (7200) -> Linear(7200 -> 10)
- KHÔNG dùng BatchNorm, KHÔNG thêm lớp Conv/FC nào khác, để giữ
  đúng tinh thần "baseline dễ kiểm soát, dễ binarize toàn bộ" mà
  thầy yêu cầu. Nếu tuần sau muốn thử BatchNorm hay thêm capacity
  cho BNN, phải làm ablation riêng (có/không) chứ không ngầm đổi
  rồi so sánh lẫn lộn với baseline này.

Cách chạy:
    python cifar10_cnn_baseline.py
    python cifar10_cnn_baseline.py --epochs 30 --batch_size 128 --lr 0.001

Sau khi baseline này chạy xong và ổn định, mọi bước sau (BNN/SNN/BSNN)
chỉ nên thay đổi ĐÚNG một thứ mỗi lần (binarize weight, hay input, hay
encoding) để giữ so sánh công bằng - đây chính là điều thầy nhấn mạnh.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 0. Vị trí thư mục chứa chính file script này
# ------------------------------------------------------------
# Dùng để neo đường dẫn output "runs_..." LUÔN nằm cùng cấp với
# file .py này, bất kể người dùng gọi lệnh `python ...` từ thư
# mục làm việc (cwd) nào. Nếu không neo theo cách này, đường dẫn
# tương đối "week05/..." sẽ được hiểu tương đối theo cwd tại thời
# điểm chạy lệnh, dẫn đến việc output có thể bị tạo ra ở một vị
# trí lồng nhau không mong muốn (ví dụ wekk05/week05/... nếu chạy
# lệnh từ bên trong thư mục wekk05/).
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. Cố định seed - điều kiện tiên quyết để so sánh công bằng
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Kiến trúc CNN baseline cho CIFAR-10
# ------------------------------------------------------------
# CỐ TÌNH giữ tối giản, bám sát ĐÚNG kiến trúc thầy đã cho ở
# baseline MNIST (Conv -> ReLU -> MaxPool -> Flatten -> Linear),
# chỉ đổi số kênh input từ 1 (ảnh xám) sang 3 (ảnh RGB) và tính
# lại kích thước Flatten cho phù hợp với ảnh 32x32.
#
#   Input 32x32x3
#   Conv(3 -> 32, 3x3, không padding)  -> 30x30x32
#   ReLU
#   MaxPool(2x2)                       -> 15x15x32
#   Flatten                            -> 32*15*15 = 7200
#   Linear(7200 -> 10)
#
# Lý do KHÔNG thêm lớp:
# - Bám sát tinh thần "baseline dễ kiểm soát" của thầy: càng ít
#   lớp, càng dễ binarize TOÀN BỘ (không sót Conv hay FC nào)
#   ở bước BNN tuần sau.
# - CIFAR-10 vốn đã khó hơn MNIST nhiều lần, nên dù chỉ 1 lớp
#   Conv, mô hình vẫn cần nhiều epoch để hội tụ thật - không lặp
#   lại lỗi "5-10 epoch đã max acc" mà thầy phê bình ở MNIST.
# - Không có BatchNorm, không có Dropout. Nếu tuần sau muốn thử
#   BatchNorm cho BNN, phải làm ablation riêng (có BN vs không
#   BN) như thầy yêu cầu, không được ngầm thêm vào rồi so sánh
#   thẳng với baseline này.
# ============================================================
class CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,   # ảnh CIFAR-10 là RGB: 3 kênh (khác MNIST 1 kênh)
            out_channels=32,
            kernel_size=3,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Conv 3x3 không padding: 32x32 -> 30x30
        # MaxPool 2x2: 30x30 -> 15x15
        # Số đặc trưng = 32 * 15 * 15 = 7200
        self.fc = nn.Linear(32 * 15 * 15, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)     # [B, 3, 32, 32] -> [B, 32, 30, 30]
        x = self.relu(x)
        x = self.pool(x)      # [B, 32, 30, 30] -> [B, 32, 15, 15]
        x = self.flatten(x)   # -> [B, 7200]
        x = self.fc(x)        # -> [B, 10], logits chưa softmax
        return x


# ============================================================
# 3. DataLoader cho CIFAR-10
# ------------------------------------------------------------
# - Chỉ dùng ToTensor() để đưa pixel về [0, 1], KHÔNG chuẩn hóa
#   theo mean/std của CIFAR-10, để giữ input ở dạng {0,1} nhất
#   quán với bước binarize input / spike encoding ở các tuần sau
#   (đúng góp ý của thầy về việc thống nhất định dạng input).
# - Chia train thành train/validation, test giữ riêng để đánh
#   giá cuối cùng.
# ============================================================
def build_dataloaders(batch_size: int, val_ratio: float, seed: int):
    transform = transforms.ToTensor()  # pixel về [0, 1], giữ nguyên 3 kênh RGB

    print("[build_dataloaders] Đang kiểm tra / tải tập train CIFAR-10 ...", flush=True)
    # Lưu ý: download=True vẫn AN TOÀN giữ nguyên dù dữ liệu đã có sẵn -
    # torchvision tự kiểm tra checksum, nếu khớp sẽ KHÔNG tải lại.
    # Chỉ cần đảm bảo thư mục ./data/cifar-10-batches-py đã tồn tại đúng vị trí.
    full_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    print(f"[build_dataloaders] Đã sẵn sàng tập train: {len(full_train)} ảnh.", flush=True)

    print("[build_dataloaders] Đang kiểm tra / tải tập test CIFAR-10 ...", flush=True)
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    print(f"[build_dataloaders] Đã sẵn sàng tập test: {len(test_dataset)} ảnh.", flush=True)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# 4. Train 1 epoch
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
# 5. Đánh giá (validation / test) - không backward, không cập nhật
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
# 6. Đếm số tham số
# ============================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================
# 7. Vẽ đồ thị train/val accuracy và loss
# ------------------------------------------------------------
# ĐÃ CHỈNH: gộp 2 biểu đồ (Accuracy, Loss) vào CHUNG 1 figure,
# đặt cạnh nhau (1 hàng x 2 cột) thay vì lưu 2 file riêng biệt
# như trước. Mục đích: nhìn 1 lần là so sánh được ngay accuracy
# và loss của cùng một lần chạy, đúng phong cách trực quan mà
# thầy muốn dùng để so sánh giữa các bước (baseline/BNN/SNN/BSNN).
#
# Cách dùng cho việc SO SÁNH nhiều model:
# - Đặt tham số `title_prefix` khác nhau cho mỗi lần chạy, ví dụ
#   "CIFAR-10 CNN Baseline", "CIFAR-10 BNN weight", ...
# - Ảnh xuất ra vẫn 1 file .png (+ 1 file .svg) duy nhất mỗi lần
#   chạy, đặt cùng thư mục out_dir như cũ.
# ============================================================
def plot_history(history, save_dir: Path, title_prefix: str = "CIFAR-10 CNN Baseline"):
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 5))

    # ---- Subplot trái: Accuracy ----
    ax_acc.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    ax_acc.plot(epochs, history["val_acc"], marker="x", label="Validation Accuracy")

    best_epoch_acc = int(np.argmax(history["val_acc"])) + 1
    best_acc = float(np.max(history["val_acc"]))
    ax_acc.axvline(best_epoch_acc, color="gray", linestyle="--", alpha=0.6)
    ax_acc.annotate(
        f"Best: {best_acc*100:.2f}%",
        xy=(best_epoch_acc, best_acc),
        xytext=(best_epoch_acc, min(history["val_acc"]) - 0.15 * (max(history["train_acc"]) - min(history["val_acc"]) + 1e-9)),
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"{title_prefix} - Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    # ---- Subplot phải: Loss ----
    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"], marker="x", label="Validation Loss")

    best_epoch_loss = int(np.argmin(history["val_loss"])) + 1
    best_loss = float(np.min(history["val_loss"]))
    ax_loss.axvline(best_epoch_loss, color="gray", linestyle="--", alpha=0.6)
    ax_loss.annotate(
        f"Best: {best_loss:.4f}",
        xy=(best_epoch_loss, best_loss),
        xytext=(best_epoch_loss + 0.3, max(history["train_loss"]) * 0.9),
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{title_prefix} - Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    fig.tight_layout()
    fig.savefig(save_dir / "cifar10_cnn_summary.png", dpi=150)
    fig.savefig(save_dir / "cifar10_cnn_summary.svg")
    plt.close(fig)


# ============================================================
# 8. Chương trình chính
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CNN baseline sạch cho CIFAR-10")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỉ lệ validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(SCRIPT_DIR / "runs_cifar10_cnn_baseline"),
        help="Thư mục lưu kết quả (mặc định: cùng cấp với file script này)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CIFAR-10 CNN BASELINE (CLEAN VERSION)")
    print(f"Device        : {device}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Seed          : {args.seed}")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    print(f"Tổng số tham số: {count_parameters(model):,}")
    print("[main] Bắt đầu vòng lặp huấn luyện ...", flush=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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

    if best_state is not None:
        torch.save(best_state, out_dir / "best_model.pt")
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("-" * 60)
    print(f"Best validation accuracy : {best_val_acc*100:.2f}% tại epoch {best_epoch}")
    print(f"Test loss                : {test_loss:.4f}")
    print(f"Test accuracy            : {test_acc*100:.2f}%")
    print(f"Checkpoint lưu tại       : {out_dir / 'best_model.pt'}")
    print("-" * 60)

    plot_history(history, out_dir, title_prefix="CIFAR-10 CNN Baseline")

    np.save(out_dir / "train_loss.npy", np.array(history["train_loss"]))
    np.save(out_dir / "train_acc.npy", np.array(history["train_acc"]))
    np.save(out_dir / "val_loss.npy", np.array(history["val_loss"]))
    np.save(out_dir / "val_acc.npy", np.array(history["val_acc"]))


if __name__ == "__main__":
    main()