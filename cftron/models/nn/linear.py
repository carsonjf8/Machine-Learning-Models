from cftron.carray import Carray
from cftron.compute_graph import comp_graph
from cftron.models.nn import Layer

import numpy as np
from typing import Callable

class Linear(Layer):
    """
    Fully connected layer.
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
        super()
        self.n_units = n_units
        self.activation_fn = activation_fn
        self.weight = None
        self.bias = None
        self.params = None
        self.use_bias = use_bias
    
    def __call__(self, data: Carray) -> Carray:
        """
        Forward pass through the layer.

        Parameters
        ----------
        data : Carray
            Input data.
        
        Returns
        -------
        Carray
            Layer output.
        """
        output = data @ self.weight
        if self.use_bias:
            output = output + self.bias
        if self.activation_fn:
            output = self.activation_fn(output)
        return output

    def _build(self, n_in: list) -> None:
        """
        Compile the layer.

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
