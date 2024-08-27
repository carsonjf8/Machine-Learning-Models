import numpy as np

def one_hot_encoder(labels: np.array) -> np.array:
    """
    One hot encodes the data. The class values are expected to be discrete and zero ordered.

    Parameters
    ----------
    labels : np.array
        Input class values.
    
    Returns
    -------
    np.array
        One hot encoded class values.
    """
    labels = np.array(labels)
    n_classes = np.max(labels) + 1
    ohe_labels = np.zeros((labels.size, n_classes))
    ohe_labels[np.arange(labels.size), labels] = 1
    return ohe_labels

def normalize(data: np.array, epsilon: float=1e-12) -> np.array:
    """
    Min-max normalizes the data array.

    Parameters
    ----------
    data : np.array
        Data to be normalized.
    epsilon : float, default = 1e-12
        Minimum value to avoid algebraic errors.
    
    Returns
    -------
    np.array
        Normalized data.
    """
    return (data - np.min(data)) / np.max((np.max(data) - np.min(data), epsilon))

def shuffle(*arrs: np.array) -> np.array:
    """
    Shuffles the data.

    Parameters
    ----------
    *arrs : np.array
        Iterable of data to be shuffled.
    
    Returns
    -------
    np.array
        Shuffled data.
    """
    output_arrs = []
    shuffle_order = np.random.randint(arrs[0].shape[0], size=arrs[0].shape[0])
    for arr in arrs:
        output_arrs.append(arr[shuffle_order])
    return output_arrs
