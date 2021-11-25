
import numpy as np
import re

def sigmoid(x_i):
    return 1 / (1 + np.exp(-x_i))

def model(X, W):
    """
    预测函数
    """
    return sigmoid(np.dot(X, W))

def cal_loss(W, X, Y, _lambda):
    size = X.shape[0]
    ln = np.mean(np.log(1 + np.exp(X @ W)))
    loss = (-Y @ X @ W + ln + 0.5 * _lambda * np.dot(W.T,W))/ size
    return loss

def cal_gradient(W, X, Y, _lambda):  
    """
    计算梯度值
    _lambda为0时即为无正则项
    """
    return ((X.T @ model(X, W) - X.T @ Y.T) + _lambda * W) / X.shape[0]

def gradient_descent(train_X, train_Y, _lambda, times, _alpha, epsilon):
    W = np.zeros((train_X.shape[1], 1))   #### 初始化 W
    new_loss = abs(cal_loss(W, train_X, train_Y, _lambda))
    k = 0
    for i in range(times):
        old_loss = new_loss
        gradient_loss = cal_gradient(W, train_X, train_Y, _lambda)
        W -= gradient_loss * _alpha   # W_i+1 = W_i - _alpha * gradient_loss
        new_loss = abs(cal_loss(W, train_X, train_Y, _lambda))
        if abs(old_loss - new_loss) < epsilon:
            k = i
            break
    return W, k

def cal_rate(test_X, test_Y, W):
    y = model(test_X, W)
    Y = np.zeros((y.shape[0], 1))
    test_Y = test_Y.T
    for i in range(y.shape[0]):
        if y[i] >= 0.5:
            Y[i] = 1
        elif y[i] < 0.5:
            Y[i] = 0
    correct = 0
    for i in range(Y.shape[0]):
        if Y[i] == test_Y[i]:
            correct += 1
    rate = correct / Y.shape[0]
    return rate

"""
W = [3*1] , X = [N*3] , Y = [1*N]
"""
if __name__ == "__main__":
    conv = 0  # 相关系数
    _lambda = 0
    times = 1000000
    _alpha = 0.5
    epsilon = 1e-6
    ### 获取数据
    bl = open("uci.data", encoding="UTF-8")
    bl_list = bl.readlines()
    LEN = len(bl_list)
    dim = 4
    X = np.zeros((LEN, dim))
    Y = np.zeros((1, LEN))
    label = []
    i = 0
    for line in bl_list:
        l = re.split("[,\n]",line)
        for j in range(dim):
            X[i, j] = l[j + 1]
        if l[0] == "L":
            Y[0, i] = 0
        elif l[0] == "R":
            Y[0, i] = 1
        i += 1
    train_X = X[:200, :]  ## 训练集
    train_Y = Y[:,: 200]
    test_X = X[200:, :]   ## 测试集
    test_Y = Y[:, 200:]
    
    ################
    W, k = gradient_descent(train_X, train_Y, _lambda, times, _alpha, epsilon)
    rate = cal_rate(test_X, test_Y, W)
    print("k = {}, lambda = {}, rate = {}".format(k, _lambda, rate))



