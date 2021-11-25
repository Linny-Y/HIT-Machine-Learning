import numpy as np

from data import *
from k_means import *
from Gmm import *

def get_data():
    with open('bezdekIris.data', 'r') as f:
        lines = [l[:-1].split(',') for l in f.readlines()]
    np.random.shuffle(lines)
    label_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    X = np.array([[float(l[i]) for i in range(len(l) - 1)] for l in lines])
    Y = np.array([label_map[l[i]] for l in lines for i in range(len(l) - 1, len(l))])
    return X, Y


if __name__ == '__main__':
    X, Y = get_data()
    center = np.zeros((3, X.shape[1]))
    C = np.zeros((X.shape[0]), dtype=np.int64)
    epoch, center, C = kmeans(3, X, C, center)
    print('k_means: epoch={}, acc={:.2f}%'.format(epoch, get_accuracy(3, Y, C)))
    epoch, C, center = GMM(3, X, Y, C, center)
    print('GMM: epoch={}, acc={:.2f}%'.format(epoch, get_accuracy(3, Y, C)))
    