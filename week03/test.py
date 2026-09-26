import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Thêm đường dẫn để import từ thư mục core
sys.path.append(os.path.abspath('..'))

# Import các thành phần cần thiết từ core
from core.utils import get_device, build_dataloaders, evaluate
from core.models import CNN

def main():
    # 1. Cấu hình đường dẫn và thiết bị
    MODEL_PATH = Path("runs_mnist_cnn/best_model.pt")
    BATCH_SIZE = 32
    VAL_RATIO = 0.1
    SEED = 42
    
    device = get_device()
    print(f"Sử dụng thiết bị cho Test: {device}")

    # 2. Xây dựng Dataloader (Lấy tập test_loader)
    # Giữ nguyên binarize_input=False và dataset="fmnist" giống file train
    _, _, test_loader = build_dataloaders(
        BATCH_SIZE, VAL_RATIO, SEED, binarize_input=False, dataset="fmnist"
    )

    # 3. Khởi tạo mô hình và nạp trọng số (.pt)
    model = CNN()
    
    if not MODEL_PATH.exists():
        print(f"❌ Lỗi: Không tìm thấy file trọng số tại {MODEL_PATH}")
        return

    # Nạp trọng số state_dict vào mô hình
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print(f" Thành công: Đã nạp trọng số từ {MODEL_PATH}")

    # 4. Đánh giá mô hình trên tập Test
    criterion = nn.CrossEntropyLoss()
    
    # Hàm evaluate trong core.utils sẽ tự động chuyển model.eval() bên trong
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "="*40)
    print(f" KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f" -> Test Loss: {test_loss:.4f}")
    print(f" -> Test Accuracy: {test_acc * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()