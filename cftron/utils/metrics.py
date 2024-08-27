import numpy as np

def minkowski_distance(x1: np.array, x2: np.array, p: int=2) -> float:
    """
    Implementation of the Minkowski distance formula.

    Parameters
    ----------
    x1: np.array
        Input data.
    x2: np.array
        Input data.
    p: float
        Exponent value. 1 for Manhattan distance, 2 for Euclidean distance, other values for custom.
    
    Returns
    -------
    float
        Minkowski distance.
    """
    return np.power(np.sum(np.power(np.subtract(x1, x2), p), axis=-1), 1/p)

def manhattan_distance(x1: np.array, x2: np.array) -> float:
    """
    Implementation of the Manhattan distance formula.

    Parameters
    ----------
    x1: np.array
        Input data.
    x2: np.array
        Input data.
    
    Returns
    -------
    float
        Manhattan distance.
    """
    return minkowski_distance(x1, x2, 1)

def euclidean_distance(x1: np.array, x2: np.array) -> float:
    """
    Implementation of the Euclidean distance formula.

    Parameters
    ----------
    x1: np.array
        Input data.
    x2: np.array
        Input data.
    
    Returns
    -------
    float
        Euclidean distance.
    """
    return minkowski_distance(x1, x2, 2)

def accuracy(y1: np.array, y2: np.array) -> float:
    """
    Implementation of the accuracy formula.

    Parameters
    ----------
    x1: np.array
        Input data.
    x2: np.array
        Input data.
    
    Returns
    -------
    float
        Accuracy.
    """
    diff = np.subtract(y1, y2)
    return 1 - (np.count_nonzero(diff) / diff.shape[0])

def entropy(y: np.array) -> float:
    """
    Implementation of the entropy formula.

    Parameters
    ----------
    y: np.array
        Input data.
    
    Returns
    -------
    float
        Entropy.
    """
    return -np.sum(np.multiply(y, np.log2(y)))

def information_gain(*args: np.array) -> float:
    """
    Implementation of the information gain formula.

    Parameters
    ----------
    *args : np.array
        Iterable of data arrays.
    
    Returns
    -------
    float
        Information gain.
    """
    all_data = np.hstack(args)

    n_data = all_data.shape[0]
    i_gain = entropy(all_data)
    for subset in args:
        i_gain -= ((subset.shape[0] / n_data) * entropy(subset))
    
    return i_gain

