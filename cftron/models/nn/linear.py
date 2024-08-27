from cftron.carray import Carray
from cftron.compute_graph import comp_graph

import numpy as np
from typing import Callable

class Linear:
    """
    Class of a fully connected layer.
    """

    def __init__(self, n_units: int, activation_fn: Callable|None=None, use_bias: bool=True):
        """
        Class constructor.

        Parameters
        ----------
        n_units : int
            Number of outputs from the layer.
        activation_fn : Callable | None, default = None
            Activation function.
        user_bias : bool, default = True
            Whether to use a bias parameter.
        """
        self.n_units = n_units
        self.activation_fn = activation_fn
        self.weight = None
        self.bias = None
        self.params = None
        self.use_bias = use_bias
    
    def __call__(self, x_batch: Carray) -> Carray:
        """
        Pass data through the layer.

        Parameters
        ----------
        x_batch : Carray
            Input data.
        
        Returns
        -------
        Carray
            Layer output.
        """
        output = x_batch @ self.weight
        if self.use_bias:
            output = output + self.bias
        if self.activation_fn:
            output = self.activation_fn(output)
        return output

    def _build(self, n_in: list) -> None:
        """
        Compiles the layer.

        Parameters
        ----------
        n_in : list
            Number of inputs to the layer.
        """
        self.weight = Carray(np.random.normal(0, 1, (n_in, self.n_units)))
        self.params = [self.weight]
        if self.use_bias:
            self.bias = Carray(np.random.normal(0, 1, (1, self.n_units)))
            self.params.append(self.bias)
    
    def _update_params(self, lr: float) -> None:
        """
        Updates the layer parameters.

        Parameters
        ----------
        lr : float
        """
        self.weight = self.weight - lr * comp_graph[self.weight.uuid]['gradient']
        if self.use_bias:
            self.bias = self.bias - lr * comp_graph[self.bias.uuid]['gradient']
