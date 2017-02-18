import numpy as np

# 二乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

