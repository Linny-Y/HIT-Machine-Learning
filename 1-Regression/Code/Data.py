############ 初始数据 ###############

import numpy as np
import matplotlib.pyplot as plt

def sin_func(x):
    return np.sin(2*np.pi*x)  # 返回初始正弦函数

def poly_func(W,x):  #多项式函数
    return x @ W


def get_point(Data_amount):
    x = np.linspace(0,1,Data_amount)  # 取Data_amount个样本
    y = sin_func(x)+np.random.normal(scale=0.05, size=x.shape)  # 加上高斯噪声
    return x, y

def get_param(Data_amount, Order):
    x_train = np.linspace(0,1,Data_amount)  # 取Data_amount个样本
    T = (sin_func(x_train)+np.random.normal(scale=0.05, size=x_train.shape)).T   # 加上高斯噪声
    X = []
    for i in range(0,Data_amount):
        x = [1.]
        for j in range(Order):
            x.append(x[-1] * x_train[i])  # X_i = [1  x_i  x_i^2  ...  x_i^M]
        X.append(x)  # X = [X_1  X_2  ...  X_i  ...  X_N]
    X = np.array(X)
    #print(X)
    return X,T

def draw_data(Data_amount):
    # sin函数
    x_sin = np.linspace(0,1,100)
    y_sin = sin_func(x_sin)
    # Data_amount个样本
    x_train, y_train = get_point(Data_amount)
    # 画图
    # plt.scatter(x_train, y_train,edgecolors="b",facecolor="none",s=40,label="Training data")
    # plt.plot(x_sin, y_sin,c="g",label="$\sin(2\pi x)$")
    # plt.legend()
    # plt.show()
    return x_sin, y_sin, x_train, y_train #返回绘图数据
#draw_data(10)
    

