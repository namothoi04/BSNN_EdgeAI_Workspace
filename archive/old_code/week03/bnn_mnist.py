import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def set_global_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.config.experimental.enable_op_determinism()

set_global_seed(42)

# -------- Custom Layer: CHỈ Binarize Weight --------
class BinaryWeightConv2D(layers.Conv2D):
    def call(self, inputs):
        # 1. Binarize weight: Ép trọng số kernel về 1.0 hoặc -1.0
        # Dùng tf.where thay vì tf.sign để tránh sinh ra giá trị 0
        w_bin = tf.where(self.kernel >= 0.0, 1.0, -1.0)
        
        # 2. Straight-Through Estimator (STE) để tính gradient
        w_bin = self.kernel + tf.stop_gradient(w_bin - self.kernel)
        
        # 3. Thực hiện phép chập (Convolution) với trọng số nhị phân
        outputs = tf.keras.backend.conv2d(
            inputs,
            w_bin,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        
        # 4. Cộng Bias (nếu có)
        if self.use_bias:
            outputs = tf.keras.backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format
            )
            
        # 5. Đi qua hàm kích hoạt (Ở đây là ReLU, CHƯA binarize)
        if self.activation is not None:
            return self.activation(outputs)
            
        return outputs

# -------- Load & Prepare MNIST --------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Chuẩn hóa về [0, 1] theo đúng chuẩn Baseline CNN ban đầu
x_train = x_train / 255.0
x_test = x_test / 255.0

# --- TÙY CHỌN: NHỊ PHÂN HÓA DỮ LIỆU ĐẦU VÀO ---
# Đổi thành True để ép ảnh đầu vào thành các pixel -1.0 và 1.0
BINARIZE_INPUT = False 
if BINARIZE_INPUT:
    print("Đang binarize dữ liệu ảnh đầu vào...")
    x_train = np.where(x_train > 0.5, 1.0, -1.0)
    x_test = np.where(x_test > 0.5, 1.0, -1.0)

# Thêm chiều kênh (vì ảnh đen trắng) -> shape: (28, 28, 1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# -------- Build Model (Giữ đúng cấu trúc Baseline) --------
model = models.Sequential([
    # Dùng lớp tự định nghĩa, CHỈ binarize weight, giữ nguyên activation='relu'
    BinaryWeightConv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# -------- Tham số huấn luyện --------
my_learning_rate = 0.001
my_epochs = 7
my_batch_size = 32

optimizer = Adam(learning_rate=my_learning_rate)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training CNN with Binary Weights (Float Activations)...")
history = model.fit(
    x_train, y_train,
    epochs=my_epochs,
    batch_size=my_batch_size,
    validation_data=(x_test, y_test)
)

# -------- Plotting --------
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

current_lr = model.optimizer.learning_rate.numpy()
info_text = f'Batch Size: {my_batch_size}\nLearning Rate: {current_lr:.4f}\nEpochs: {my_epochs}\nInput Binarized: {BINARIZE_INPUT}'

plt.figure(figsize=(15, 7))

# --- Biểu đồ Accuracy ---
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='x')
plt.title('Binary Weight CNN - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='center left')
plt.grid(True)
plt.text(0.5, 0.5, info_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# --- Biểu đồ Loss ---
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x')
plt.title('Binary Weight CNN - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='center left')
plt.grid(True)
plt.text(0.5, 0.5, info_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='center', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()