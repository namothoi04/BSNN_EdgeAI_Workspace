import numpy as np

# 1. Khởi tạo cấu trúc mạng (Weights & Bias)
def initialize_parameters(n_x, n_h, n_y):
    # n_x: số neuron input, n_h: số neuron hidden, n_y: số neuron output
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return W1, b1, W2, b2

# 2. Hàm kích hoạt (Sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. Quá trình Lan truyền tiến (Forward Propagation)
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1) # Dùng Tanh cho hidden layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2) # Dùng Sigmoid cho output layer (phân loại nhị phân)
    return Z1, A1, Z2, A2

# 4. Quá trình Lan truyền ngược (Backward Propagation)
def backward_propagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[1] # Số lượng mẫu dữ liệu
    
    # Tính đạo hàm từ output layer ngược về
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Tính đạo hàm cho hidden layer
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) # Đạo hàm của Tanh
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

# 5. Cập nhật trọng số (Gradient Descent)
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2
