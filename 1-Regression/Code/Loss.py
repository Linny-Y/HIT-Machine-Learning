############## 计算误差 #############

import numpy as np
import matplotlib.pyplot as plt
import math
 
from Data import *

def cal_loss(X, W, T, _lambda):  #### 计算误差
    return 0.5 * ((X @ W - T).T @ (X @ W - T) + (10 ** _lambda) * W.T @ W)

################### 无正则项 ########################
def get_loss_with_Order():  # 阶数对损失的影响
    Data_amount = 10
    Order = []
    Loss = []
    for order in range(15):
        Order.append(order)
        X, T = get_param(Data_amount, order)
        W = np.linalg.pinv(X) @ T
        loss = 0.5 * (X @ W - T).T @ (X @ W - T)
        Loss.append(loss)
    
    plt.title("Loss of different order")
    plt.xlabel("Order")
    plt.ylabel("Loss")
    plt.plot(Order, Loss,c="b")
    plt.show()

def get_loss_with_DataAmount():
    Order = 10
    Data_amount = []
    Loss = []
    for amount in range(5, 20, 1) :
        Data_amount.append(amount)
        X, T = get_param(amount, Order)
        W = np.linalg.pinv(X) @ T
        loss = 0.5 * (X @ W - T).T @ (X @ W - T)
        Loss.append(loss)
    
    plt.title("Loss of different data amout")
    plt.xlabel("Data amount")
    plt.ylabel("Loss")
    plt.plot(Data_amount, Loss,c="b")
    plt.show()
################### 有正则项 ########################
def get_loss_with_lambda():
    Order = 10
    Data_amount = 10
    Loss = []
    ln_lambda = -30
    ln = []
    while ln_lambda <= 0:
        ln.append(ln_lambda)
        X, T = get_param(Data_amount, Order)
        W = np.linalg.pinv(X) @ T
        loss = cal_loss(X, W, T, ln_lambda)
        Loss.append(loss)
        ln_lambda += 1
    
    plt.title("Loss of different lambda")
    plt.xlabel("$ln\lambda$")
    plt.ylabel("Loss")
    plt.plot(ln, Loss,c="b")
    plt.show()


# get_loss_with_Order()
# get_loss_with_DataAmount()
# get_loss_with_lambda()