from cftron.carray import Carray
from cftron.models.nn.activation import log_softmax

def mse(y_pred: Carray, y_actual: Carray) -> Carray:
    """
    Implementation of the mean square error loss function.

    Parameters
    ----------
    y_pred : Carray
        Predicted values.
    y_actual : Carray
        Ground truth values.
    
    Returns
    -------
    Carray
        Loss value.
    """
    dif = y_pred - y_actual
    sqr = Carray.power(dif, 2)
    mean = Carray.mean(sqr)
    return mean

def mae(y_pred: Carray, y_actual: Carray) -> Carray:
    """
    Implementation of the mean average error loss function.

    Parameters
    ----------
    y_pred : Carray
        Predicted values.
    y_actual : Carray
        Ground truth values.
    
    Returns
    -------
    Carray
        Loss value.
    """
    dif = y_pred - y_actual
    mean = Carray.mean(dif)
    return mean

def softmax_cross_entropy(y_pred: Carray, y_actual: Carray) -> Carray:
    """
    Implementation of softmax combined with cross entropy loss function.

    Parameters
    ----------
    y_pred : Carray
        Class predicted logit values.
    y_actual : Carray
        One hot encoded ground truth class values.
    
    Returns
    -------
    Carray
        Loss value.
    """
    return -Carray.mean(Carray.sum(log_softmax(y_pred) * y_actual, axis=1))

def cross_entropy(y_pred_sm: Carray, y_actual: Carray) -> Carray:
    """
    Implementation of the cross entropy loss function.

    Parameters
    ----------
    y_pred : Carray
        Class predicted probability values.
    y_actual : Carray
        One hot encoded ground truth class values.
    
    Returns
    -------
    Carray
        Loss value.
    """
    return -Carray.mean(Carray.sum(y_pred_sm * y_actual, axis=1))
