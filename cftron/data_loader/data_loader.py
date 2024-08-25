import numpy as np

from cftron.carray import Carray
from cftron import utils
class DataLoader:
    def __init__(self, x_data, y_data, batch_size, shuffle=False, normalize=False, one_hot_encode=False):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.batch_size = batch_size
        
        if shuffle:
            self.x_data, self.y_data = utils.shuffle(self.x_data, self.y_data)
        if normalize:
            self.x_data = utils.normalize(self.x_data)
        if one_hot_encode:
            self.y_data = utils.one_hot_encoder(self.y_data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        x_batch = self.x_data[self.index * self.batch_size : (self.index + 1) * self.batch_size]
        y_batch = self.y_data[self.index * self.batch_size : (self.index + 1) * self.batch_size]

        if x_batch.shape[0] == 0:
            raise StopIteration
        
        self.index += 1
        return (
            Carray(x_batch),
            Carray(y_batch)
        )
    
    def __len__(self):
        return int(np.ceil(self.x_data.shape[0] / self.batch_size))
    
    def size(self):
        return self.x_data.shape[0]
