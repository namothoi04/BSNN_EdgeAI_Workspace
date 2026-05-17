import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def seed_everything(seed: int = 42) -> None: 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*13*13, 10)
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def build_dataloaders(batch_size, val_ratio, seed):
    transform = transforms.ToTensor()
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

    val_size = int(len(full_train)*val_ratio)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
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
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct+= (preds == labels).sum().item()
        total+= labels.size(0)
    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc
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
        loss = criterion(logits,labels)

        total_loss+=loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct+= (preds == labels).sum().item()
        total+= labels.size(0)
    avg_loss = total_loss/total
    avg_acc = correct/total
    return avg_loss, avg_acc

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def plot_history(history, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], marker="x", label="Validation Accuracy")
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc = float(np.max(history["val_acc"]))
    plt.annotate(
        f"Best Val: {best_acc*100:.2f}% @ epoch {best_epoch}",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch, best_acc)
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST CNN Baseline - Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "mnist_cnn_acc.png", dpi=150)
    plt.savefig(save_dir / "mnist_cnn_acc.svg")
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="x", label="Validation Loss")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_loss = float(np.min(history["val_loss"]))
    plt.annotate(
        f"Best Val Loss: {best_loss:.4f} @ epoch {best_epoch}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch, best_loss)
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST CNN Baseline - Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "mnist_cnn_loss.png", dpi=150)
    plt.savefig(save_dir / "mnist_cnn_loss.svg")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="CNN baseline tối giản cho MNIST")
    parser.add_argument("--dataset", type=str, default="mnist", help="chọn dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỉ lệ validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập")
    parser.add_argument("--out_dir", type=str, default="runs_mnist_cnn_clean", help="Thư mục lưu kết quả")
    args = parser.parse_args()

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

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Mô hình
    model = CNN().to(device)
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

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,device 
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion,device 
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



    
