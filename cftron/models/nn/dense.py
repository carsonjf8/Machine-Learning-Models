from cftron.carray import Carray
from cftron.compute_graph import comp_graph

import numpy as np

class Dense:
    def __init__(self, n_units, activation_fn, use_bias=True) -> None:
        self.n_units = n_units
        self.activation_fn = activation_fn
        self.weight = None
        self.bias = None
        self.params = None
        self.use_bias = use_bias
    
    def __call__(self, x_batch):
        output = x_batch @ self.weight
        if self.use_bias:
            output = output + self.bias
        if self.activation_fn:
            output = self.activation_fn(output)
        return output

    def _build(self, n_in: list) -> None:
        self.weight = Carray(np.random.normal(0, 1, (n_in, self.n_units)))
        self.params = [self.weight]
        if self.use_bias:
            self.bias = Carray(np.random.normal(0, 1, (1, self.n_units)))
            self.params.append(self.bias)
    
    def _update_params(self, learning_rate):
        self.weight = self.weight - learning_rate * comp_graph[self.weight.uuid]['gradient']
        if self.use_bias:
            self.bias = self.bias - learning_rate * comp_graph[self.bias.uuid]['gradient']
