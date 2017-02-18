import sys, os
sys.path.append(os.pardir)
import numpy as np
from chapter_04.gradient_method import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


net = simpleNet()
# 重みパラメータの表示
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print("max index is ", np.argmax(p))

# 正解ラベルの導出
t = np.zeros(3, dtype=int)
t[np.argmax(p)] = 1

net.loss(x, t)

# 勾配を求める
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)