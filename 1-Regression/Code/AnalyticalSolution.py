################ 多项式拟合 #################

import numpy as np
import matplotlib.pyplot as plt

from Data import *


################ 无正则项 ##################
def get_poly(Data_amount, Order):  
    X, T = get_param(Data_amount, Order)
    W = np.linalg.pinv(X) @ T  # 多项式函数系数 W
    #print(W.shape, T.shape)
    X_test, _ = get_param(100, Order)  # 多项式函数数据 X_test
    x = np.linspace(0,1,100)
    y = poly_func(W,X_test)
    return x, y   # 获取多项式函数x,y
    
def draw_poly(Data_amount):
    x_sin, y_sin, x_train, y_train = draw_data(Data_amount)  
    plt.figure(figsize=(9,8))
    for i,order in enumerate([0, 1, 3, 9]):  
        plt.subplot(2,2,i+1)

        x, y = get_poly(Data_amount,order)
        
        plt.scatter(x_train, y_train,edgecolors="b",facecolor="none",s=40,label="Training data")
        plt.plot(x_sin, y_sin,c="g",label="$\sin(2\pi x)$")
        plt.plot(x, y,c="r",label="Fitting curve")
        plt.title("M={}".format(order))
        plt.legend()
    plt.show()

################ 有正则项 ##################
def get_poly_with_penalty(Data_amount, Order, _lambda):  
    X, T = get_param(Data_amount, Order)
    W = np.linalg.pinv(X.T @ X + _lambda * np.identity(X.shape[1])) @ X.T @ T  # 多项式函数系数 W
    
    #print(W.shape)
    X_test, _ = get_param(100, Order)  # 多项式函数数据 X_test
    x = np.linspace(0,1,100)
    y = poly_func(W,X_test)
    return x, y   # 获取多项式函数x,y
    

def draw_poly_with_penalty(Data_amount, _lambda):
    x_sin, y_sin, x_train, y_train = draw_data(Data_amount)  
    plt.figure(figsize=(9,8))
    for i,order in enumerate([0, 1, 3, 9]):  
        plt.subplot(2,2,i+1)
        print(x_train.shape)
        x, y = get_poly_with_penalty(Data_amount,order, _lambda)
        
        plt.scatter(x_train, y_train,edgecolors="b",facecolor="none",s=40,label="Training data")
        plt.plot(x_sin, y_sin,c="g",label="$\sin(2\pi x)$")
        plt.plot(x, y,c="r",label="Fitting curve")
        plt.title("M={}".format(order))
        plt.legend()
    plt.show()

#draw_poly(10)  # 无正则项
#draw_poly_with_penalty(10)   # 有正则项