"""
SNN-Temporalcoding.py

SNN baseline cho MNIST sử dụng Temporal (Latency) Coding.
- Kiến trúc: Conv -> MaxPool -> LIF -> Flatten -> Linear -> LIF
- Encoding: Latency Coding (Pixel sáng bắn xung sớm)
- Decoding: Membrane Potential Accumulation
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

# Khai báo snnTorch
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate


# ============================================================
# 1. Hàm cố định seed
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
# 3. Định nghĩa mô hình SNN (Temporal)
# ============================================================
class SimpleSNN_Temporal(nn.Module):
    def __init__(self, beta: float = 0.9) -> None:
        super().__init__()

        # Sử dụng Surrogate Gradient
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Khối trích xuất đặc trưng
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Khối phân loại
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 13 * 13, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x_steps: torch.Tensor):
        num_steps = x_steps.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = [] 

        for step in range(num_steps):
            x_t = x_steps[step]

            # Layer 1
            cur1 = self.pool(self.conv1(x_t))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc(self.flatten(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# ============================================================
# 4. Tạo DataLoader
# ============================================================
def build_dataloaders(batch_size: int, val_ratio: float, seed: int):
    transform = transforms.ToTensor()

    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# 5. Hàm train 1 epoch (Tích hợp Latency Encoding)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, num_steps, tau_latency):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # bypass=True để bỏ qua xung trễ, tránh lỗi Out of bounds
        spike_data = spikegen.latency(images, num_steps=num_steps, tau=tau_latency, threshold=0.01, bypass=True)

        optimizer.zero_grad()
        
        spk_rec, mem_rec = model(spike_data) 
        
        # Temporal Decoding: Tổng điện thế màng
        logits = mem_rec.sum(dim=0) 

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
# 6. Hàm đánh giá
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_steps, tau_latency):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        spike_data = spikegen.latency(images, num_steps=num_steps, tau=tau_latency, threshold=0.01, bypass=True)
        spk_rec, mem_rec = model(spike_data)
        
        logits = mem_rec.sum(dim=0)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def plot_history(history, save_dir: Path, model_name = 'SNN - Temporal Coding', file_name = 'history_temporal'):
    acc_title = model_name + ' - Accuracy'
    loss_title = model_name + ' - Loss'
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
    ax1.set_title(acc_title)
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
    ax2.set_title(loss_title)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / (file_name+'.png'), dpi=150)
    plt.savefig(save_dir / (file_name+'.svg'))
    plt.close()


# ============================================================
# 9. Chương trình chính
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SNN sử dụng Temporal (Latency) Coding cho MNIST")
    parser.add_argument("--epochs", type=int, default=7, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỉ lệ validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập")
    parser.add_argument("--out_dir", type=str, default="runs_mnist_snn_temporalcoding", help="Thư mục lưu kết quả")
    
    # Tham số đặc thù cho SNN
    parser.add_argument("--num_steps", type=int, default=25, help="Số bước thời gian (Time steps)")
    parser.add_argument("--beta", type=float, default=0.9, help="Hệ số rò rỉ màng điện thế")
    parser.add_argument("--tau_latency", type=float, default=5.0, help="Hằng số thời gian cho hàm mã hóa độ trễ")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    
    # Đảm bảo lưu vào cùng thư mục với file chạy
    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MNIST SNN BASELINE (TEMPORAL CODING)")
    print(f"Device      : {device}")
    print(f"Time steps  : {args.num_steps}")
    print(f"Epochs      : {args.epochs}")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_dataloaders(args.batch_size, args.val_ratio, args.seed)

    model = SimpleSNN_Temporal(beta=args.beta).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    print(f"Total parameters: {count_parameters(model):,}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.num_steps, args.tau_latency)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.num_steps, args.tau_latency)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # In terminal giống hệt Rate Coding
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save(best_state, out_dir / "best_model.pt")
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, args.num_steps, args.tau_latency)
    
    # In test giống hệt Rate Coding
    print("-" * 60)
    print(f"Test accuracy           : {test_acc*100:.2f}%")
    print("-" * 60)

    plot_history(history, out_dir, "SNN - temporal coding", "history_temporal")

if __name__ == "__main__":
    main()