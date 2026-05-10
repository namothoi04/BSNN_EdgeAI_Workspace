# %%
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Kiểm tra xem TensorFlow đã sẵn sàng chưa
print("TensorFlow version:", tf.__version__)

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#cung khong can thiet reshape   
x_train.reshape(-1, 28, 28, 1)
x_test.reshape(-1, 28, 28, 1)
#normalize: đưa về giá trị [0,1] cho dễ tính toán
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)


# %%
plt.imshow(x_train[100], cmap='gray') 
plt.title(f"Nhãn thực tế: {y_train[100]}") 
plt.show() 

# %%
#build model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)


# %%
test_loss, test_accuracy = model.evaluate(x_test, y_test)



