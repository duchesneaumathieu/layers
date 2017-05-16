import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *

class ShapeSplitLayer(lasagne.layers.Layer):
    def __init__(self, incoming, idx=0, **kwargs):
        super(ShapeSplitLayer, self).__init__(incoming, **kwargs)
        self.idx=idx
    
    def get_output_for(self, input, **kwargs):
        return input[self.idx]
    
    def get_output_shape_for(self, shape, **kwargs):
        return (shape[self.idx],)

class ShapeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(ShapeLayer, self).__init__(incoming, **kwargs)
        self.ndims = len(self.input_shape)
        
    def get_output_for(self, input, **kwargs):
        return input.shape
    
    def get_output_shape_for(self, shape, **kwargs):
        return shape
    
    def split(self):
        return [ShapeSplitLayer(self, n) for n in range(self.ndims)]

class FeatureMapsShapeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, divide=1, multiply=1, **kwargs):
        super(FeatureMapsShapeLayer, self).__init__(incoming, **kwargs)
        self.divide_h = divide if isinstance(divide, int) else divide[0]
        self.divide_w = divide if isinstance(divide, int) else divide[1]
        
        self.multiply_h = multiply if isinstance(multiply, int) else multiply[0]
        self.multiply_w = multiply if isinstance(multiply, int) else multiply[1]
    
    def get_output_for(self, input, **kwargs):
        bs, fs, h, w = input.shape
        return (h*self.multiply_h)//self.divide_h, (w*self.multiply_w)//self.divide_w
    
    def get_output_shape_for(self, shape, **kwargs):
        #Not typical: return the modified 
        #shape of the input not the shape of the output
        bs, fs, h, w = shape
        h_ = None if h is None else (h*self.multiply_h)//self.divide_h
        w_ = None if w is None else (w*self.multiply_w)//self.divide_w
        return h_, w_
    
class ReshapeLayer2(lasagne.layers.MergeLayer):
    #inputs: [Layer, ShapeLayer], outputs: in0[in1]
    def get_output_for(self, inputs, **kwargs):
        return inputs[0].reshape(inputs[1])
    
    def get_output_shape_for(self, shapes, **kwargs):
        return shapes[1]