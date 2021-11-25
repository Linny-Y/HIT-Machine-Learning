import numpy as np

from data import *

def kmeans(k, X, C, center, epochs=50, epsilon=1e-7):
    """
    K-means算法实现，算得分类结果和中心
    """
    # 初始化均值向量
    for i in range(k):  
        center[i] = X[i]
    distance = np.zeros(k)
    epoch = 0
    # 开始迭代
    for t in range(epochs):
        # 计算距离，划分进距离最小得簇
        for j in range(X.shape[0]):
            for i in range(k):
                distance[i] = np.linalg.norm(X[j]-center[i])
            C[j] = np.argmin(distance) # 划分至距离最短处
        # 计算新的均值向量
        count = np.zeros((k),dtype=np.int64)
        new_center = np.zeros((k, X.shape[1]))
        for j in range(X.shape[0]):
            new_center[C[j]] += X[j]
            count[C[j]] += 1  # 簇中元素个数
        for i in range(k):
            new_center[i] = new_center[i] / count[i]  
        # 判断是否符合结束迭代条的件
        if(np.linalg.norm(new_center - center) < epsilon):
            center = new_center
            epoch = t
            break
        # 更新均值向量 
        center = new_center
    return epoch, center, C # 返回迭代次数 均值向量 簇划分



if __name__ == "__main__":
    k = 3
    size = 150
    loc = np.array([[2, 7], [6, 2], [8, 7]])
    cov = np.array([[[1, 0], [0, 2]], [[2, 0], [0, 1]], [[1, 0], [0, 1]]])
    X, Y, C, center = data_generator(k, size, loc, cov)
    # print(X.shape[0], X.shape[1], Y.shape[0]) # 450, 2   450
    epoch, center, C = kmeans(k, X, C, center)
    draw(k, X, Y, C, center, epoch)


    


