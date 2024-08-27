from cftron.carray import Carray

def relu(arr: Carray) -> Carray:
    """
    Implementation of the ReLU activation function.

    Parameters
    ----------
    arr : Carray
        Input data.
    
    Returns
    -------
    Carray
        Output data.
    """
    return Carray.clip(arr, a_min=0, a_max=None)

def log_softmax(arr: Carray) -> Carray:
    """
    Implementation of the log softmax activation function.

    Parameters
    ----------
    arr : Carray
        Input data.
    
    Returns
    -------
    Carray
        Output data.
    """
    arr -= Carray.expand_dims(Carray.max(arr, axis=1), axis=1)
    return arr - Carray.expand_dims(Carray.log(Carray.sum(Carray.exp(arr), axis=1)), axis=1)
