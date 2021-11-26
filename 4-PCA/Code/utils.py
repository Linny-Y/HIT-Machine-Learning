import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image

def pca(x, k):
    """
        pca降维
    """
    mu = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    values, vectors = np.linalg.eig(cov)
    index = np.argsort(values)[: -(k + 1): -1]  # 取最大的 k 个的下标值
    vectors = vectors[:, index]  # 取对应下标的特征向量
    x_pca = (x - mu).dot(vectors) + mu  # 重建数据
    return x_pca

def generate_data(mu, sigma, n=100):
    """
    生成高斯分布数据
    """
    x = np.random.multivariate_normal(mean=mu, cov=sigma, size=n)
    return x






if __name__ == '__main__':
    print(' ')