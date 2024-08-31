from typing import Any
from cftron.carray import Carray
from cftron.compute_graph import comp_graph

class Layer:
    """
    Base class of all layers.
    """

    def __init__(self):
        """
        Class constructor.
        """
        self.params = []

    def __call__(self, data: Carray) -> Carray:
        """
        Forward pass through the layer. This function should be overriden by subclasses.

        Parameters
        ----------
        data : Carray
            Input data to the layer.
        
        Raises
        -------
        NotImplementedError
            If the function is not overriden by subclasses.
        """
        raise NotImplementedError

    def _build(self, n_in: int) -> None:
        """
        Compile the layer. This function should be overriden by subclasses.

        Parameters
        ----------
        n_in : int
            Input size to the layer.
        
        Raises
        -------
        NotImplementedError
            If the function is not overriden by subclasses.
        """
        raise NotImplementedError

    def _update_params(self, lr: float) -> None:
        """
        Update the layer parameters.

        Parameters
        ----------
        lr : float
        """
        for i in range(len(self.params)):
            self.params[i] = self.params[i] - lr * self.params[i].grad
