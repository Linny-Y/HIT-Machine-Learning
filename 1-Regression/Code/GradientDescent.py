#################### 梯度下降 ################

import numpy as np
import matplotlib.pyplot as plt

from Data import *
from Loss import *

def cal_gradient(X, W, T, _lambda):  ####  计算梯度值
    return X.T @ X @ W - X.T @ T + (10 ** _lambda) * W

def gradient_descent(Data_amount, Order, _lambda, times, _alpha, epsilon):
    X, T = get_param(Data_amount, Order)

    new_W = np.zeros((Order + 1))   #### 初始化 W
    new_loss = abs(cal_loss(X, new_W, T, _lambda))
    k = 0
    for i in range(times):
        old_loss = new_loss
        gradient_loss = cal_gradient(X, new_W, T, _lambda)
        old_W = new_W
        new_W -= gradient_loss * _alpha   # W_i+1 = W_i - _alpha * gradient_loss
        new_loss = abs(cal_loss(X, new_W, T, _lambda))
        if old_loss < new_loss: #不下降了，说明步长过大
            new_W = old_W
            _alpha /= 2
        if old_loss - new_loss < epsilon:
            k = i
            break
    X_test, _ = get_param(100, Order)  # 多项式函数数据 X_test
    x = np.linspace(0,1,100)
    y = poly_func(new_W,X_test)
    return x, y, k

