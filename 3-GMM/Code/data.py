import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# 获取数据
def data_generator(k, size, loc, cov):
    X = np.zeros((k * size, loc.shape[1]))
    Y = np.zeros((k * size), dtype=np.int64)
    for i in range(k):
        X[i*size: (i+1)*size] = np.random.multivariate_normal(loc[i], cov[i], size)
        Y[i*size: (i+1)*size] = i
    center = np.zeros((k, loc.shape[1]))
    C = np.zeros((k * size), dtype=np.int64)
    shuffle = np.random.permutation(X.shape[0])
    X, Y = X[shuffle], Y[shuffle]
    # draw(k, X, Y, Y, loc, 0)
    return X, Y, C, center

# 计算准确度
def get_accuracy(k, Y, C):
    max_acc = 0
    classes = list(permutations(range(k), k)) # 获取分类的全排列
    for i in range(len(classes)):
        acc = 0
        for j in range(len(Y)):
            if(Y[j] == classes[i][C[j]]):
                acc += 1
        acc /= len(Y)
        # print(acc)
        max_acc = max(max_acc, acc)
    return max_acc * 100

# 画图
def draw(k, X, Y, C, center, epoch):
    colors = ['c','y', 'g', 'b']
    for i in range(X.shape[0]):
        plt.plot(X[i, 0], X[i, 1], '.', c = colors[C[i]])
    plt.scatter(center[:,0],center[:,1], marker = 'x', c = 'r')
    plt.title('epoch={}, acc={:.2f}%'.format(epoch, get_accuracy(k, Y, C)))
    plt.show()

