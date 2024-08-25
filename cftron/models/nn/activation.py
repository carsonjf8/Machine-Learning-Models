from ...carray import Carray

def relu(arr: Carray) -> Carray:
    return Carray.clip(arr, a_min=0, a_max=None)

'''
def softmax(arr: Carray) -> Carray:
    arr -= Carray.expand_dims(Carray.max(arr, axis=1), axis=1)
    numerator = Carray.exp(arr)
    denominator = Carray.expand_dims(Carray.sum(Carray.exp(arr), axis=1), axis=1)
    return numerator / denominator
'''

def log_softmax(arr: Carray) -> Carray:
    arr -= Carray.expand_dims(Carray.max(arr, axis=1), axis=1)
    return arr - Carray.expand_dims(Carray.log(Carray.sum(Carray.exp(arr), axis=1)), axis=1)
