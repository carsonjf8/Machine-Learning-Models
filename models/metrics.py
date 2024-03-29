import numpy as np

'''
implementation of the Minkowski distance formula
returns Minkowski distance
x1: NumPy Array
    input data
x2: NumPy Array
    input data
p: int, default=2
    exponent value, 1 for Manhattan distance, 2 for Euclidean distance, other values for custom
'''
def minkowski_distance(x1, x2, p=2):
    return np.power(np.sum(np.power(np.subtract(x1, x2), p), axis=-1), 1/p)

'''
implementation of the Manhattan distance formula
returns Manhattan distance
x1: NumPy Array
    input data
x2: NumPy Array
    input data
'''
def manhattan_distance(x1, x2):
    return minkowski_distance(x1, x2, 1)

'''
implementation of the Euclidean distance formula
returns Euclidean distance
x1: NumPy Array
    input data
x2: NumPy Array
    input data
'''
def euclidean_distance(x1, x2):
    return minkowski_distance(x1, x2, 2)

'''
implementation of accuracy formula
returns accuracy
y1: NumPy Array
    input data
y2: NumPy Array
    input data
'''
def accuracy(y1, y2):
    diff = np.subtract(y1, y2)
    return 1 - (np.count_nonzero(diff) / diff.shape[0])
    