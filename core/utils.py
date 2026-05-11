#utilities
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import glob
import os
import cv2
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def build_dataloaders(batch_size, val_ratio, seed, binarize_input=False):
    transform_list = [transforms.ToTensor()]
    
    if binarize_input:
        print("Đang áp dụng: Binarize Input Transform")
        transform_list.append(BinarizeTransform())
        
    transform = transforms.Compose(transform_list)

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
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
       
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / total, correct / total

#visualization
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

class BinarizeTransform:
    def __call__(self, x):
        return torch.where(x > 0.5, torch.tensor(1.0), torch.tensor(-1.0))

def predict_external_images(model, folder_path, device):
    """
    Hàm load các ảnh từ folder, tiền xử lý và dự đoán nhãn.
    """
    model.eval()
    image_paths = glob.glob(os.path.join(folder_path, "*.*")) # Lấy tất cả file ảnh
    
    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong thư mục: {folder_path}")
        return

    print(f"\n--- Đang dự đoán ảnh từ thư mục: {folder_path} ---")
    
    with torch.no_grad():
        for img_path in image_paths:
            # 1. Đọc ảnh và chuyển về ảnh xám (Grayscale)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 2. Resize về 28x28 (kích thước của MNIST)
            img_resized = cv2.resize(img, (28, 28))
            
            # 3. Tiền xử lý: Chuyển sang tensor, thêm dimension (Batch, Channel, H, W) và chuẩn hóa
            # MNIST thường là nền đen chữ trắng, nếu ảnh của bạn nền trắng chữ đen thì cần đảo ngược màu:
            # img_resized = 255 - img_resized 
            
            img_tensor = torch.from_numpy(img_resized).float().to(device)
            img_tensor = img_tensor / 255.0  # Chuẩn hóa về [0, 1]
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 28, 28]

            # 4. Dự đoán
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
            
            print(f"Ảnh: {os.path.basename(img_path)} --> Dự đoán: {pred}")


