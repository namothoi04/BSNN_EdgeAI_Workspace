

import sys
import os
import cv2


sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import từ thư mục core
from core.utils import seed_everything, get_device, build_dataloaders, train_one_epoch, evaluate, count_parameters, plot_history
from core.models import BNN

# Configs
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
VAL_RATIO = 0.1
SEED = 42
OUT_DIR = Path("runs_mnist_bnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

seed_everything(SEED)
device = get_device()
print(f"Sử dụng thiết bị: {device}")


train_loader, val_loader, test_loader = build_dataloaders(
    BATCH_SIZE, VAL_RATIO, SEED, binarize_input=False 
)
#
model = BNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Tổng tham số BNN: {count_parameters(model):,}")

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

if best_state is not None:
    torch.save(best_state, OUT_DIR / "best_model.pt")
    model.load_state_dict(best_state)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print("-" * 40)
print(f"Test accuracy: {test_acc*100:.2f}%")


plot_history(history, OUT_DIR)

np.save(OUT_DIR / "train_loss.npy", np.array(history["train_loss"]))
np.save(OUT_DIR / "train_acc.npy", np.array(history["train_acc"]))
np.save(OUT_DIR / "val_loss.npy", np.array(history["val_loss"]))
np.save(OUT_DIR / "val_acc.npy", np.array(history["val_acc"]))





