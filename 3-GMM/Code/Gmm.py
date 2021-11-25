import numpy as np
from scipy.stats import multivariate_normal

from data import *

def GMM(k, X, Y, C, center, epochs=100, epsilon=1e-7):
    # 初始化高斯混合成分概率
    _alpha = np.ones(k) * (1.0 / k)
    # 初始化均值向量
    center = X[:k]
    # print(center)
    # 初始化协方差矩阵
    _sigma = np.array([0.1 * np.eye(X.shape[1])] * k)
    # print(_sigma[0])
    likelihood = 0
    for t in range(epochs):
        # 计算后验概率 E步
        _gamma = cal_gamma(k, X, _alpha, _sigma, center)
        
        # print(_gamma[0])
        # 划分簇
        C = np.argmax(_gamma, axis=1)
        # M步 计算新参数值
        new_alpha = cal_alpha(_gamma, _alpha)
        new_sigma = cal_sigma(X, _gamma, center, _sigma)
        new_center = cal_center(X, _gamma, center)
        # 计算极大对数似然
        new_likelihood = log_likelihood(X, center, _sigma, _alpha)
        # 判断退出条件
        if abs(new_likelihood - likelihood) < epsilon:
            epoch = t
            break
        _alpha = new_alpha
        _sigma = new_sigma
        center = new_center
        likelihood = new_likelihood
    return epoch, C, center

def cal_gamma(k, X, _alpha, _sigma, _mu, epsilon=1e-7):
    _gamma = np.zeros((X.shape[0], k))
    # print(_sigma)
    for i in range(X.shape[0]):
        sum = 0
        p = 0
        for j in range(k):
            p = multivariate_normal.pdf(X[i], mean=_mu[j], cov=_sigma[j])
            if abs(p) < epsilon:
                p += epsilon
            _gamma[i][j] = _alpha[j] * p
            sum += _gamma[i][j]
        for j in range(k):
            _gamma[i][j] /= sum
    return _gamma    

def cal_alpha(_gamma, _alpha):
    m = _gamma.shape[0]
    k = _gamma.shape[1]
    for i in range(k):
        for j in range(m):
            _alpha[i] += _gamma[j][i]
        _alpha[i] /= m
    return _alpha

def cal_center(X, _gamma, center):
    m = _gamma.shape[0]
    k = _gamma.shape[1]
    for i in range(k):
        sum = 0
        for j in range(m):
            sum += _gamma[j][i]
            center[i] += _gamma[j][i] * X[j]
        center[i] /= sum
    return center

def cal_sigma(X, _gamma, center, _sigma):
    _sigma = np.zeros_like(_sigma)
    m = _gamma.shape[0]
    k = _gamma.shape[1]
    for i in range(k):
        sum = 0
        for j in range(m):
            sum += _gamma[j][i]
            v = (X[j]-center[i]).reshape(-1, 1)
            _sigma[i] += _gamma[j][i] * np.dot(v,v.T)
        _sigma[i] /= sum
    return _sigma

def log_likelihood(X, _mu, _sigma, _alpha):
    """
    计算极大似然对数
    """
    log = 0
    for i in range(X.shape[0]):
        pi_times_pdf_sum = 0
        for j in range(_mu.shape[0]):
            pi_times_pdf_sum += _alpha[j] * multivariate_normal.pdf(X[j], mean=_mu[j], cov=_sigma[j])
        log += np.log(pi_times_pdf_sum)
    return log

if __name__ == '__main__':
    k = 3
    size = 150
    loc = np.array([[2, 7], [6, 2], [8, 7]])
    cov = np.array([[[1, 0], [0, 2]], [[2, 0], [0, 1]], [[1, 0], [0, 1]]])
    X, Y, center, C = data_generator(k, size, loc, cov)
    epoch, C, center = GMM(k, X, Y, C, center)
    draw(k, X, Y, C, center, epoch)



