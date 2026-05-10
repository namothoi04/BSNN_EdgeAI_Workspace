import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------- Set Global Seed --------
def set_global_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(42)

# Thiết lập thiết bị chạy (GPU nếu có, ngược lại là CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------- Load & Prepare MNIST --------
BINARIZE_INPUT = False 

# Khởi tạo các phép biến đổi ảnh
transform_list = [transforms.ToTensor()] # Đã bao gồm việc chuẩn hóa về [0, 1]

if BINARIZE_INPUT:
    print("Đang binarize dữ liệu ảnh đầu vào...")
    class BinarizeTransform:
        def __call__(self, x):
            return torch.where(x > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
    transform_list.append(BinarizeTransform())

transform = transforms.Compose(transform_list)

# Tải dữ liệu
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# -------- Tham số huấn luyện --------
my_learning_rate = 0.001
my_epochs = 7
my_batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=my_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=my_batch_size, shuffle=False)

# -------- Build Model (Baseline CNN chuẩn) --------
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # SỬ DỤNG nn.Conv2d CHUẨN THAY VÌ LỚP NHỊ PHÂN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Tính toán kích thước đầu vào cho lớp Dense (Linear)
        self.fc1 = nn.Linear(32 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)        
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x             

model = BaselineCNN().to(device)

# -------- Optimizer & Loss --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=my_learning_rate)

# Lịch sử huấn luyện để vẽ biểu đồ
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

# -------- Training Loop --------
print("Training BASELINE CNN (Float Weights & Float Activations)...")
for epoch in range(my_epochs):
    # --- Huấn luyện ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    history['loss'].append(epoch_loss)
    history['accuracy'].append(epoch_acc)
    
    # --- Đánh giá (Validation) ---
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_val_loss = val_loss / total
    epoch_val_acc = correct / total
    history['val_loss'].append(epoch_val_loss)
    history['val_accuracy'].append(epoch_val_acc)
    
    print(f"Epoch {epoch+1}/{my_epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f} - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f}")

# -------- Plotting --------
epochs_range = range(1, my_epochs + 1)
info_text = f'Batch Size: {my_batch_size}\nLearning Rate: {my_learning_rate:.4f}\nEpochs: {my_epochs}\nInput Binarized: {BINARIZE_INPUT}'

plt.figure(figsize=(15, 7))

# --- Biểu đồ Accuracy ---
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(epochs_range, history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Baseline CNN - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='center left')
plt.grid(True)
plt.text(0.5, 0.5, info_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# --- Biểu đồ Loss ---
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['loss'], label='Training Loss', marker='o')
plt.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='x')
plt.title('Baseline CNN - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='center left')
plt.grid(True)
plt.text(0.5, 0.5, info_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='center', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()