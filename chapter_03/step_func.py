import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

## step_function section
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
# y軸の範囲を指定
plt.ylim(-0.1, 1.1)
plt.show()

## sigmoid section
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
# y軸の範囲を指定
plt.ylim(-0.1, 1.1)
plt.show()

## relu section
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
# y軸の範囲を指定
plt.ylim(-0.1, 5.0)
plt.show()
