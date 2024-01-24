import numpy as np

def minkowski_distance(x1, x2, p=2):
    return np.power(np.sum(np.power(np.subtract(x1, x2), p), axis=-1), p)

def manhattan_distance(x1, x2):
    return minkowski_distance(x1, x2, 1)

def euclidean_distance(x1, x2):
    return minkowski_distance(x1, x2, 2)

def accuracy(y1, y2):
    diff = np.subtract(y1, y2)
    return 1 - (np.count_nonzero(diff) / diff.shape[0])
    