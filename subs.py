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