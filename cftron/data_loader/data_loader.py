from cftron.carray import Carray
from cftron import utils
from cftron.data_loader import DataLoader

import numpy as np

class DataLoader:
    """
    Class containing data during a training loop.
    """
    
    def __init__(self,
                 x_data: np.array|list,
                 y_data: np.array|list,
                 batch_size: int,
                 shuffle: bool=False,
                 normalize: bool=False,
                 one_hot_encode: bool=False):
        """
        Class constructor.

        Parameters
        ----------
        x_data : np.array | list
            Input data passed to the model.
        y_data : np.array | list
            Ground truth expected output from the model.
        batch_size : int
            Number of data examples passed to the model at a time.
        shuffle : bool, default = False
            Whether to shuffle the data rows or not.
        normalize : bool, default = False
            Whether to normalize the input data or not.
        one_hot_encode : bool, default = False
            Whether to one hot encode the output data or not.
        """
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.batch_size = batch_size
        
        if shuffle:
            self.x_data, self.y_data = utils.shuffle(self.x_data, self.y_data)
        if normalize:
            self.x_data = utils.normalize(self.x_data)
        if one_hot_encode:
            self.y_data = utils.one_hot_encoder(self.y_data)

    def __iter__(self) -> DataLoader:
        """
        Returns the iterator of the data loader.

        Returns
        -------
        DataLoader
            Iterator.
        """
        self.index = 0
        return self

    def __next__(self) -> tuple[Carray, Carray]:
        """
        Gets the next element from the iterator.

        Returns
        -------
        tuple[Carray, Carray]
            Tuple of input and expected output data.
        """
        x_batch = self.x_data[self.index * self.batch_size : (self.index + 1) * self.batch_size]
        y_batch = self.y_data[self.index * self.batch_size : (self.index + 1) * self.batch_size]

        if x_batch.shape[0] == 0:
            raise StopIteration
        
        self.index += 1
        return (
            Carray(x_batch),
            Carray(y_batch)
        )
    
    def __len__(self) -> int:
        """
        Returns the length of the data loader, the number of batches.

        Returns
        -------
        int
            Number of batches in the data loader.
        """
        return int(np.ceil(self.x_data.shape[0] / self.batch_size))
    
    def size(self) -> int:
        """
        Returns the number of elements in the dataset.

        Returns
        -------
        int
            Number of data points in the dataset.
        """
        return self.x_data.shape[0]
