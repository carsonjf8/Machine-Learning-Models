import numpy as np
import uuid

from ..compute_graph import comp_graph, ComputationGraph

class Carray:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.uuid = str(uuid.uuid4())
    
    def shape(self):
        return self.data.shape
    
    def size(self):
        return self.data.size
    
    def grad(self):
        return comp_graph.graph[self.uuid]['gradient']
    
    def __str__(self):
        return f'Carray {self.shape()}\n{self.data}'
    
    def backward(self):
        comp_graph.graph[self.uuid]['gradient'] = np.ones_like(self.data)
        comp_graph.backward(self.uuid)

    def __neg__(self):
        result = Carray(-self.data)
        comp_graph.add_operation(ComputationGraph.Operation.NEG, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data))
        return result
    
    def __add__(self, arr):
        result = Carray(self.data + arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.ADD, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __radd__(self, arr):
        result = Carray(arr.data + self.data)
        comp_graph.add_operation(ComputationGraph.Operation.ADD, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result

    def __sub__(self, arr):
        result = Carray(self.data - arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.SUBTRACT, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __rsub__(self, arr):
        result = Carray(arr.data - self.data)
        comp_graph.add_operation(ComputationGraph.Operation.SUBTRACT, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result

    def __mul__(self, arr):
        result = Carray(self.data * arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.MULTIPLY, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __rmul__(self, arr):
        result = Carray(arr.data * self.data)
        comp_graph.add_operation(ComputationGraph.Operation.MULTIPLY, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result

    def __truediv__(self, arr):
        result = Carray(self.data / arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.DIVIDE, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __rtruediv__(self, arr):
        result = Carray(arr.data / self.data)
        comp_graph.add_operation(ComputationGraph.Operation.DIVIDE, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result

    def __matmul__(self, arr):
        if self.data.ndim != 2 and arr.data.ndim != 2:
            raise ValueError('Matrix is not 2-dimensional')
        result = Carray(self.data @ arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.MATMUL, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __rmatmul__(self, arr):
        result = Carray(arr.data @ self.data)
        comp_graph.add_operation(ComputationGraph.Operation.MATMUL, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result
    
    def clip(arr, a_min, a_max):
        result = Carray(np.clip(arr.data, a_min=a_min, a_max=a_max))
        op_args = {'a_min':a_min, 'a_max':a_max}
        comp_graph.add_operation(ComputationGraph.Operation.CLIP, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def exp(arr):
        result = Carray(np.exp(arr.data))
        comp_graph.add_operation(ComputationGraph.Operation.EXP, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data))
        return result
    
    def log(arr):
        result = Carray(np.log(arr.data))
        comp_graph.add_operation(ComputationGraph.Operation.LOG, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data))
        return result
    
    def power(arr, exp):
        result = Carray(np.power(arr.data, exp))
        op_args = {'exp':exp}
        comp_graph.add_operation(ComputationGraph.Operation.POWER, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def sum(arr, axis=None):
        result = Carray(np.sum(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.SUM, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def mean(arr, axis=None):
        return Carray.sum(arr, axis=axis) / Carray(arr.size() if axis == None else arr.shape()[axis])
    
    def max(arr, axis=None):
        result = Carray(np.max(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.MAX, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def min(arr, axis=None):
        result = Carray(np.min(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.MIN, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def expand_dims(arr, axis):
        result = Carray(np.expand_dims(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.EXPAND_DIMS, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
        
    def squeeze(arr, axis):
        result = Carray(np.squeeze(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.SQUEEZE, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    