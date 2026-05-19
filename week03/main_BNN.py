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
#setup seed
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#choose cpu or gpu
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#creat Binary weight
class BinaryWeightConv2d(nn.Conv2d):
    def forward(self, input):
        w_bin = torch.where(self.weight >= 0,
                            torch.tensor(1.0, device=self.weight.device),
                            torch.tensor(-1.0, device=self.weight.device)
                            )
        w_bin = self.weight + (w_bin - self.weight).detach()
        return F.conv2d(input, w_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)

#Binarize input to test acc

class BinarizeTransform:
        def __call__(self, x):
            return torch.where(x > 0.5, torch.tensor(1.0), torch.tensor(-1.0))

#Binary activation function
class BinaryActivation(nn.Module):
    def forward(self, x):
        out = torch.where(x >= 0, 
                        torch.tensor(1.0, device=x.device),
                        torch.tensor(-1.0, device=x.device))
        return x + (out - x).detach()
#BNN model
class BNN(nn.Module):
    def __init__(self, activation_type:str= "binary"):
        super(BNN, self).__init__()
        
        self.conv1 = BinaryWeightConv2d(1, 32, 3)
        self.bn =  nn.BatchNorm2d(32)

        if activation_type == "binary":
            self.act = BinaryActivation()
        elif activation_type == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Chỉ hỗ trợ 'binary' hoặc 'relu'")
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*13*13, 10)
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
def build_dataloaders(batch_size, val_ratio, seed, binarize_input=False, dataset = "mnist"):
    transform_list = [transforms.ToTensor()]
    
    if binarize_input:
        print("Đang áp dụng: Binarize Input Transform")
        transform_list.append(BinarizeTransform())
        
    transform = transforms.Compose(transform_list)
    if (dataset == "mnist"):
        full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif (dataset == "fmnist"):
        full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)


    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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

    # Khởi tạo 1 Figure chứa 2 Subplots (1 hàng, 2 cột)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ==========================================
    # ĐỒ THỊ 1: ACCURACY (BÊN TRÁI)
    # ==========================================
    ax1.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    ax1.plot(epochs, history["val_acc"], marker="x", label="Validation Accuracy")
    
    best_epoch_acc = int(np.argmax(history["val_acc"])) + 1
    best_acc = float(np.max(history["val_acc"]))
    
    ax1.axvline(x=best_epoch_acc, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(
        f"Best: {best_acc*100:.2f}%",
        xy=(best_epoch_acc, best_acc),
        xytext=(best_epoch_acc - 1.5, best_acc - 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5)
    )
    
    ax1.set_title("MNIST BNN - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ==========================================
    # ĐỒ THỊ 2: LOSS (BÊN PHẢI)
    # ==========================================
    ax2.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax2.plot(epochs, history["val_loss"], marker="x", label="Validation Loss")
    
    best_epoch_loss = int(np.argmin(history["val_loss"])) + 1
    best_loss = float(np.min(history["val_loss"]))
    
    ax2.axvline(x=best_epoch_loss, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(
        f"Best: {best_loss:.4f}",
        xy=(best_epoch_loss, best_loss),
        xytext=(best_epoch_loss - 1.5, best_loss + 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5)
    )
    
    ax2.set_title("MNIST BNN - Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ==========================================
    # LƯU ẢNH
    # ==========================================
    plt.tight_layout() # Tự động căn chỉnh khoảng cách để không bị lẹm chữ
    plt.savefig(save_dir / "mnist_bnn_history_combined.png", dpi=150)
    plt.savefig(save_dir / "mnist_bnn_history_combined.svg")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="BNN MNIST")
    parser.add_argument("--dataset", type=str, default="mnist", help="chọn dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỉ lệ validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập")
    parser.add_argument("--out_dir", type=str, default="runs_mnist_BNN", help="Thư mục lưu kết quả")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MNIST BNN")
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
        binarize_input=False,
        dataset=args.dataset
    )

    # Mô hình
    model = BNN(activation_type="relu").to(device)
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

    # Lưu lịch sử để tiện xem lại
    np.save(out_dir / "train_loss.npy", np.array(history["train_loss"]))
    np.save(out_dir / "train_acc.npy", np.array(history["train_acc"]))
    np.save(out_dir / "val_loss.npy", np.array(history["val_loss"]))
    np.save(out_dir / "val_acc.npy", np.array(history["val_acc"]))


if __name__ == "__main__":
    main()



    
