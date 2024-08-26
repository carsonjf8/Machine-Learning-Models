from cftron.compute_graph import comp_graph, ComputationGraph
from cftron.carray import Carray

import numpy as np
import uuid

class Carray:
    """
    Array class.
    """

    def __init__(self, data: list | np.array):
        """
        Class constructor.

        Parameters
        ----------
        data : list | np.array
            Values contained in the array.
        """
        self.data = np.array(data, dtype=np.float32)
        self.uuid = str(uuid.uuid4())
    
    def shape(self) -> tuple[int, ...]:
        """
        Gets the shape of the array.

        Returns
        -------
        tuple[int, ...]
            Shape of the array containing the data.
        """
        return self.data.shape
    
    def size(self) -> int:
        """
        Gets the number of elements in the array.

        Returns
        -------
        int
            Number of elements in the array.
        """
        return self.data.size
    
    def grad(self) -> np.array:
        """
        Gets the gradient of the array with respect ot the final value.

        Returns
        -------
        np.array
            Gradient of the array.
        """
        return comp_graph.graph[self.uuid]['gradient']
    
    def __str__(self) -> str:
        """
        Gets the string representation of the array.

        Returns
        -------
        str
            String representation of the array.
        """
        return f'Carray {self.shape()}\n{self.data}'
    
    def backward(self) -> None:
        """
        Computes the gradient of the array with respect to the final value.
        """
        comp_graph.graph[self.uuid]['gradient'] = np.ones_like(self.data)
        comp_graph.backward(self.uuid)

    def __neg__(self) -> Carray:
        """
        Applies the negative operation to the data.

        Returns
        -------
        Carray
            Negative result.
        """
        result = Carray(-self.data)
        comp_graph.add_operation(ComputationGraph.Operation.NEG, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data))
        return result
    
    def __add__(self, arr: Carray) -> Carray:
        """
        Applies the addition operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to add with.
        
        Returns
        -------
        Carray
            Result of summed arrays.
        """
        result = Carray(self.data + arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.ADD, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __radd__(self, arr: Carray) -> Carray:
        """
        Applies the right addition operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to add with.
        
        Returns
        -------
        Carray
            Result of summed arrays.
        """
        result = Carray(arr.data + self.data)
        comp_graph.add_operation(ComputationGraph.Operation.ADD, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result

    def __sub__(self, arr: Carray) -> Carray:
        """
        Applies the subtraction operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to subtract with.
        
        Returns
        -------
        Carray
            Result of difference of arrays.
        """
        result = Carray(self.data - arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.SUBTRACT, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __rsub__(self, arr: Carray) -> Carray:
        """
        Applies the right subtraction operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to subtract with.
        
        Returns
        -------
        Carray
            Result of difference of arrays.
        """
        result = Carray(arr.data - self.data)
        comp_graph.add_operation(ComputationGraph.Operation.SUBTRACT, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result

    def __mul__(self, arr: Carray) -> Carray:
        """
        Applies the multiplication operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to multiply with.
        
        Returns
        -------
        Carray
            Result of product of arrays.
        """
        result = Carray(self.data * arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.MULTIPLY, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __rmul__(self, arr: Carray) -> Carray:
        """
        Applies the right multiplication operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to multiply with.
        
        Returns
        -------
        Carray
            Result of product of arrays.
        """
        result = Carray(arr.data * self.data)
        comp_graph.add_operation(ComputationGraph.Operation.MULTIPLY, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result

    def __truediv__(self, arr: Carray) -> Carray:
        """
        Applies the division operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to divide with.
        
        Returns
        -------
        Carray
            Result of quotient of arrays.
        """
        result = Carray(self.data / arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.DIVIDE, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __rtruediv__(self, arr: Carray) -> Carray:
        """
        Applies the right division operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to divide with.
        
        Returns
        -------
        Carray
            Result of quotient of arrays.
        """
        result = Carray(arr.data / self.data)
        comp_graph.add_operation(ComputationGraph.Operation.DIVIDE, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result

    def __matmul__(self, arr: Carray) -> Carray:
        """
        Applies the matrix multiplication operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to matrix multiply with.
        
        Returns
        -------
        Carray
            Result of matrix multiplication of arrays.
        """
        if self.data.ndim != 2 and arr.data.ndim != 2:
            raise ValueError('Matrix is not 2-dimensional')
        result = Carray(self.data @ arr.data)
        comp_graph.add_operation(ComputationGraph.Operation.MATMUL, result.uuid, np.copy(result.data),
                                 self.uuid, np.copy(self.data), uuid_2=arr.uuid, data_2=np.copy(arr.data))
        return result
    def __rmatmul__(self, arr: Carray) -> Carray:
        """
        Applies the right matrix multiplication operation to the data.

        Parameters
        ----------
        arr : Carray
            Other array to matrix multiply with.
        
        Returns
        -------
        Carray
            Result of matrix multiplication of arrays.
        """
        result = Carray(arr.data @ self.data)
        comp_graph.add_operation(ComputationGraph.Operation.MATMUL, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), uuid_2=self.uuid, data_2=np.copy(self.data))
        return result
    
    def clip(arr: Carray, a_min: float, a_max: float) -> Carray:
        """
        Applies the clip operation to the data.

        Parameters
        ----------
        arr : Carray
            Array to clip the data.
        a_min : float
            Minimum value to clip the data to.
        a_max : float
            Maximum value to clip the data to.
        
        Returns
        -------
        Carray
            Result of clipped data.
        """
        result = Carray(np.clip(arr.data, a_min=a_min, a_max=a_max))
        op_args = {'a_min':a_min, 'a_max':a_max}
        comp_graph.add_operation(ComputationGraph.Operation.CLIP, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def exp(arr: Carray) -> Carray:
        """
        Applies the exponential operation to the data.

        Parameters
        ----------
        arr : Carray
            Array to apply the exponential function to.
        
        Returns
        -------
        Carray
            Result of the exponential of the data.
        """
        result = Carray(np.exp(arr.data))
        comp_graph.add_operation(ComputationGraph.Operation.EXP, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data))
        return result
    
    def log(arr: Carray) -> Carray:
        """
        Applies the logarithmic operation to the data.

        Parameters
        ----------
        arr : Carray
            Array to apply the logarithmic function to.
        
        Returns
        -------
        Carray
            Result of the logarithmic of the data.
        """
        result = Carray(np.log(arr.data))
        comp_graph.add_operation(ComputationGraph.Operation.LOG, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data))
        return result
    
    def power(arr: Carray, exp: float) -> Carray:
        """
        Raises the data to a power.

        Parameters
        ----------
        arr : Carray
            Input data.
        exp : float
            Power ot raise the data to.
        
        Returns
        -------
        Carray
            Result of raising the data to a power.
        """
        result = Carray(np.power(arr.data, exp))
        op_args = {'exp':exp}
        comp_graph.add_operation(ComputationGraph.Operation.POWER, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def sum(arr: Carray, axis: int|None=None) -> Carray:
        """
        Applies the sum operation to the data.

        Parameters
        ----------
        arr : Carray
            Array to apply the sum function to.
        axis : int
            Axis to sum across.
        
        Returns
        -------
        Carray
            Result of the summed data.
        """
        result = Carray(np.sum(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.SUM, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def mean(arr: Carray, axis: int|None=None) -> Carray:
        """
        Applies the mean operation to the data.

        Parameters
        ----------
        arr : Carray
            Array to apply the mean function to.
        axis : int
            Axis to average across.
        
        Returns
        -------
        Carray
            Result of the averaged data.
        """
        return Carray.sum(arr, axis=axis) / Carray(arr.size() if axis == None else arr.shape()[axis])
    
    def max(arr: Carray, axis: int|None=None) -> Carray:
        """
        Applies the max operation to the data.

        Parameters
        ----------
        arr : Carray
            Array to apply the max function to.
        axis : int
            Axis to calculate the max on.
        
        Returns
        -------
        Carray
            Result of the max operation.
        """
        result = Carray(np.max(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.MAX, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def min(arr: Carray, axis: int|None=None) -> Carray:
        """
        Applies the min operation to the data.

        Parameters
        ----------
        arr : Carray
            Array to apply the min function to.
        axis : int
            Axis to calculate the min on.
        
        Returns
        -------
        Carray
            Result of the min operation.
        """
        result = Carray(np.min(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.MIN, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    
    def expand_dims(arr: Carray, axis: int) -> Carray:
        """
        Expands the dimensions of the array.

        Parameters
        ----------
        arr : Carray
            Array to expand the dimension of.
        axis : int
            Axis to expand the dimension of.
        
        Returns
        -------
        Carray
            Array with expanded dimensions.
        """
        result = Carray(np.expand_dims(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.EXPAND_DIMS, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
        
    def squeeze(arr: Carray, axis: int) -> Carray:
        """
        Reduce the dimensions of the array.

        Parameters
        ----------
        arr : Carray
            Array to reduce the dimension of.
        axis : int
            Axis to reduce the dimension of.
        
        Returns
        -------
        Carray
            Array with reduced dimensions.
        """
        result = Carray(np.squeeze(arr.data, axis=axis))
        op_args = {'axis':axis}
        comp_graph.add_operation(ComputationGraph.Operation.SQUEEZE, result.uuid, np.copy(result.data),
                                 arr.uuid, np.copy(arr.data), op_args=op_args)
        return result
    