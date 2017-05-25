import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *

class SubLayer(lasagne.layers.Layer):
    def __init__(self, incoming, idx=0, **kwargs):
        super(SubLayer, self).__init__(incoming, **kwargs)
        self.idx=idx
        
    def get_output_for(self, input, **kwargs):
        return input[self.idx]
    
    def get_output_shape_for(self, shape, **kwargs):
        return shape[1:]
    
class OutputSplitLayer(lasagne.layers.Layer):
    def __init__(self, incoming, idx=0, **kwargs):
        super(OutputSplitLayer, self).__init__(incoming, **kwargs)
        self.idx=idx
    
    def get_output_for(self, input, **kwargs):
        return input[self.idx]
    
    def get_output_shape_for(self, shape, **kwargs):
        return shape[self.idx]

from theano_ops import unravel_index_Op
class ProjectionLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, axis=1, **kwargs):
        """
        A, B = incomings
        A.ndims must be equal to B.ndims+1
        B.shape must be equal to A.shape without "axis"
        
        e.g. #1
        if axis = 0,
        A = [[00,01,02], [10,11,12]],
        B = [1, 0 ,1]
        then, Indexing(A, B) = [10, 01, 12]
        
        e.g. #2
        if axis = 1,
        A = [[[000, 001], [010, 011], [020, 021]], [[100, 101], [110, 111], [120, 121]]],
        B = [[1, 2], [2, 0]]
        then, Indexing(A, B) = [[010, 021], [120, 101]]
        """
        super(ProjectionLayer, self).__init__(incomings, **kwargs)
        self.axis = axis
        
    def get_output_for(self, inputs, **kwargs):
        A, B = inputs
        shape = B.shape
        ravel = unravel_index_Op(T.arange(T.prod(shape)), shape)
        tupled_ravel = tuple(ravel[i] for i in range(B.ndim))
        total = tupled_ravel[:self.axis] + (B.flatten(),) + tupled_ravel[self.axis:]
        return A[total].reshape(shape)
    
    def get_output_shape_for(self, inputs, **kwargs):
        return inputs[1]