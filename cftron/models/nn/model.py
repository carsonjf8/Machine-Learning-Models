from cftron.carray import Carray
from cftron.compute_graph import comp_graph

class Model:
    """
    Generic class used for creating a model with multiple layers.
    """
    
    def __init__(self, input_shape: tuple[int, ...] | int, layers, learning_rate):
        """
        Class constructor.

        Parameters
        ---------
        input_shape : tuple[int, ...] | int
            Shape of input data to the model.
        """
        
        self.input_shape = input_shape
        self.layers = layers
        self.learning_rate = learning_rate
    
    def __call__(self, x_batch: Carray) -> Carray:
        """
        Passes data through the model to produce output.

        Parameters
        ----------
        x_batch : Carray
            Input data.
        
        Returns
        -------
        Carray
            Model output.
        """
        x = x_batch
        for layer in self.layers:
            x = layer(x)
        return x
    
    def build(self) -> None:
        """
        Compiles the model.
        """
        for index, layer in enumerate(self.layers):
            layer._build(self.layers[index - 1].n_units if index > 0 else self.input_shape)
    
    def update_weights(self) -> None:
        """
        Updates the weights of the model.
        """
        for layer in self.layers:
            for param in layer.params:
                param.data = param.data - self.learning_rate * comp_graph.graph[param.uuid]['gradient']
