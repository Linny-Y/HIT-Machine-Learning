################### 共轭梯度 ################

import numpy as np
import matplotlib.pyplot as plt

from Data import *

def conjugate_gradient(X, T, Order, _lambda, epsilon, times):
    A = np.transpose(X) @ X - (10 ** _lambda) * np.identity(len(np.transpose(X)))
    b = np.transpose(X) @ T
    x = np.random.normal(size=(A.shape[1]))
    r_0 = A @ x - b
    p = -r_0
    k = times
    for i in range(times):
        _alpha = (r_0.T @ r_0) / (p.T @ A @ p)
        x = x + _alpha * p
        r = r_0 + _alpha * A @ p
        if (r_0.T @ r_0) < epsilon:
            k = i
            break
        _beta = r.T @ r / (r_0.T @ r_0)
        p = -r + _beta * p
        r_0 = r
    X_test, _ = get_param(100, Order)  
    x_ = np.linspace(0,1,100)
    y = poly_func(x,X_test)
    return x_, y, k







