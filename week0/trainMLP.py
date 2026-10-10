import numpy as np
np.random.seed(42)

class Linear:
    def __init__(self, in_features, out_features): #dimension of input and output vector X(1, in)*W(in, out) + b(1, out) = Y(1, out)
        self.weights = np.random.rand(in_features, out_features)*np.sqrt(2.0/ in_features)
        self.bias = np.zeros((1, out_features))
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias
    def backward(self, dout): #dL/dW = X^T*dout
        #gradients
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.weights.T)
        return dx

class ReLU:
    def __init__(self):
        self.x = None
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    def backward(self, dout):
        #d(ReLU)/dx = dout/dx = 1 or 0 -> dx
        dx = dout*(self.x>0) 
        return dx
class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true)**2)
    def backward(self):
        batch_size = self.y_pred.shape[0]
        return (2.0 / batch_size)*(self.y_pred - self.y_true)

#generate synthetic dataset
X = np.linspace(-2, 2, num=100).reshape(-1, 1) #100 real numbers from -2 to 2
y_true = X**2

# Instantiate network layers
# Architecture: Input(1) -> Hidden (8) -> ReLU -> Output Layer(1)
hidden_layer = Linear(in_features=1, out_features=8)
activation = ReLU()
output_layer = Linear(in_features=8, out_features=1)
loss_fn = MSELoss()

# Hyperparameters
learning_rate = 0.05
epochs = 1000

#training loop
for epoch in range(epochs):
    # forward
    out1 = hidden_layer.forward(X)
    out2 = activation.forward(out1)
    y_pred = output_layer.forward(out2)

    loss = loss_fn.forward(y_pred, y_true)

    # backward
    d_loss = loss_fn.backward()
    d_out2 = output_layer.backward(d_loss)
    d_out1 = activation.backward(d_out2)
    _ = hidden_layer.backward(d_out1)

    # gradient descent step (parameter update)
    # update output layer weights and biases
    output_layer.weights -= learning_rate*output_layer.dW
    output_layer.bias -= learning_rate*output_layer.db

    #update hidden layer weights and biases
    hidden_layer.weights -= learning_rate*hidden_layer.dW
    hidden_layer.bias -= learning_rate*hidden_layer.db

    if (epoch+1)%200 == 0:
        print(f"Epoch [{epoch+1}]/[{epochs}], Loss: {loss:.6f}")
    
#test
test_x = np.array([[1.5], [-1.0]])
predicted = output_layer.forward(
    activation.forward(
        hidden_layer.forward(test_x)
    )
)
print("predict:")
for x_val, p_val, true_val in zip(test_x, predicted, test_x**2):
    print(f"Input: {x_val[0]:.2f} | Predicted: {p_val[0]:.4f} | Actual (x^2): {true_val[0]:.4f}")