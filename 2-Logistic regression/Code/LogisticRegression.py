
import numpy as np
import matplotlib.pyplot as plt

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

def data(size, locat, conv):
    cov=[[0.4, conv], [conv, 0.4]]
    def generater(n):
        X = np.zeros((n * 2, 2))
        Y = np.zeros((n * 2, 1))
        X[:n, :] = np.random.multivariate_normal(locat[0], cov, n)
        X[n:, :] = np.random.multivariate_normal(locat[1], cov, n)
        Y[n:] = 1
        return X, Y.T
    train_data = generater(size[0])
    test_data = generater(size[1])
    return train_data, test_data

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
    size = (100,150) 
    locat = np.array([[1, 3],[2, 5]])
    conv = 0  # 相关系数
    _lambda = 0
    times = 1000000
    _alpha = 0.5
    epsilon = 1e-6
    ### 生成数据
    train_data, test_data = data(size, locat, conv)
    train_X = np.zeros((size[0] * 2, 3))
    train_X[:,:2] = train_data[0]
    train_X[:,2:] = 1
    train_Y = train_data[1]

    test_X = np.zeros((size[1] * 2, 3))
    test_X[:,:2] = test_data[0]
    test_X[:,2:] = 1
    test_Y = test_data[1]
    ################
    W, k = gradient_descent(train_X, train_Y, _lambda, times, _alpha, epsilon)
    rate = cal_rate(test_X, test_Y, W)

    x = np.linspace(-1, 6, 10000)
    y = -(W[0] * x + W[2]) / W[1]
    A = []
    B = []
    for i in range(size[0] * 2):
        if train_data[1][0][i] == 0:
            A.append(train_data[0][i])
        elif train_data[1][0][i] == 1:
            B.append(train_data[0][i])
    A = np.array(A)
    B = np.array(B)
    plt.title("k = {}, conv = {}, $\lambda$ = {}, rate = {:.2f}".format(k, conv, _lambda, rate))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y, c="b",label = "Train")
    plt.scatter(A[:,:1], A[:, 1:], c="r")
    plt.scatter(B[:,:1], B[:, 1:], c="g")
    plt.legend()
    plt.show()



