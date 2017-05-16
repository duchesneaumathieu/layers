import numpy as np
import theano
import theano.tensor as T
import lasagne

class SoftMaxLayer(lasagne.layers.Layer):
    def __init__(self, incoming, axis=1, **kwargs):
        super(SoftMaxLayer, self).__init__(incoming, **kwargs)
        self.axis = axis
        
    def get_output_for(self, input, **kwargs):
        exp = T.exp(input-T.max(input, axis=self.axis, keepdims=True))
        return exp/exp.sum(self.axis, keepdims=True)

class ConstantLayer(lasagne.layers.InputLayer):
    def __init__(self, constant, **kwargs):
        super(ConstantLayer, self).__init__(constant.shape, **kwargs)
        constant = np.array(constant, dtype=theano.config.floatX)
        self.shape = constant.shape
        self.input_var = theano.shared(constant)
        
class TheanoWrapperLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, fn, shape_fn, **kwargs):
        super(TheanoWrapperLayer, self).__init__(incomings, **kwargs)
        self.fn = fn
        self.shape_fn = shape_fn
        
    def get_output_for(self, inputs, **kwargs):
        return self.fn(*inputs)
    
    def get_output_shape_for(self, shapes, **kwargs):
        return self.shape_fn(*shapes)
        
class ZerosShapeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, shape, **kwargs):
        super(ZerosShapeLayer, self).__init__(incoming, **kwargs)
        self.shape = shape
        
    def get_output_for(self, input, **kwargs):
        shape = []
        sh = self.shape
        insh = input.shape
        for i in range(len(self.shape)):
            shape.append(insh[i] if sh[i]=='x' else sh[i])
        return T.zeros(shape)
    
    def get_output_shape_for(self, input_shape, **kwargs):
        shape = []
        sh = self.shape
        insh = input_shape
        for i in range(len(self.shape)):
            shape.append(insh[i] if sh[i]=='x' else sh[i])
        return tuple(shape)
    
class OnesShapeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, shape, **kwargs):
        super(OnesShapeLayer, self).__init__(incoming, **kwargs)
        self.shape = shape
        
    def get_output_for(self, input, **kwargs):
        shape = []
        sh = self.shape
        insh = input.shape
        for i in range(len(self.shape)):
            shape.append(insh[i] if sh[i]=='x' else sh[i])
        return T.ones(shape)
    
    def get_output_shape_for(self, input_shape, **kwargs):
        shape = []
        sh = self.shape
        insh = input_shape
        for i in range(len(self.shape)):
            shape.append(insh[i] if sh[i]=='x' else sh[i])
        return tuple(shape)

from theano_ops import munkresOp
class MunkresLayer(lasagne.layers.Layer):
    def get_output_for(self, a, **kwargs):
        #inputs: a.shape=(bs,sq,sq)
        return munkresOp(-a) #used for maximization
    
    def get_output_shape_for(self, shape, **kwargs):
        return shape[:2]
    
from theano_ops import permutationOp
class PermutationLayer(lasagne.layers.MergeLayer):
    def get_output_for(self, inputs, **kwargs):
        vs, ps = inputs
        return permutationOp(vs, ps)
        
    def get_output_shape_for(self, shapes, **kwargs):
        return shapes[0]
    
from theano_ops import assignmentMaskOp
class AssignmentMeanLayer(lasagne.layers.MergeLayer):
    def get_output_for(self, inputs, **kwargs):
        inp, idx = inputs
        mask = assignmentMaskOp(idx)
        return (inp * mask).sum(axis=1).mean()
    
    def get_output_shape_for(self, shapes, **kwargs):
        return ()