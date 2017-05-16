import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.init as init
import lasagne.utils as utils

from lasagne.layers import *
from collections import OrderedDict

class NormalisationLayer(lasagne.layers.Layer):
    def __init__(self, incoming, axes='auto', alpha=0.1, epsilon=1e-4,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(NormalisationLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.alpha = alpha
        self.epsilon = epsilon

        # create parameters, ignoring all dimensions in axes
        self.shape = [1 if axis in axes else size
                      for axis, size in enumerate(self.input_shape)]
        if any(size is None for size in self.shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, self.shape, 'beta',
                trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, self.shape, 'gamma',
                trainable=True, regularizable=True)
            
        self.mean = self.add_param(mean, self.shape, 'mean',
            trainable=False, regularizable=False, batch_norm_stat=True)
        self.inv_std = self.add_param(inv_std, self.shape, 'inv_std',
            trainable=False, regularizable=False, batch_norm_stat=True)
        
    def get_output_for(self, input, training=False, updates=OrderedDict(), **kwargs):
        if training:
            mean = input.mean(self.axes, keepdims=True)
            inv_std = T.inv(T.sqrt(input.var(self.axes, keepdims=True) + self.epsilon))
            updates[self.mean] = (1-self.alpha)*self.mean + self.alpha*mean
            updates[self.inv_std] = (1-self.alpha)*self.inv_std + self.alpha*inv_std
        else:
            mean = self.mean
            inv_std = self.inv_std
            
        return (input - mean)*inv_std*self.gamma + self.beta
    
class AdamLayer(lasagne.layers.Layer):
    def __init__(
        self, incoming, learning_rate=0.01,
        beta1=0.9, beta2=0.999, epsilon=1e-08, clipping=5, **kwargs):
        super(AdamLayer, self).__init__(incoming, **kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clipping = clipping
        
        self.all_params = lasagne.layers.get_all_params(self.input_layer, trainable=True)
        
        self.define_params()
        
    def define_params(self):
        self.t_prev = self.add_param(
            theano.shared(utils.floatX([0.])), (1,), 't_prev',
            trainable=False, regularizable=False)
        
        self.m_prevs = list()
        self.v_prevs = list()
        for param in self.all_params:
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            self.m_prevs.append(self.add_param(
                    m_prev, value.shape, 'm_prev', trainable=False, regularizable=False))
            self.v_prevs.append(self.add_param(
                    v_prev, value.shape, 'v_prev', trainable=False, regularizable=False))
            
    def gradient_clipping(self, steps):
        if self.clipping == 0: return steps
        
        norm = T.sqrt(T.sum([T.sum(step**2) for step in steps]))
        renorm = T.min([self.clipping, norm])/(norm+1e-7)
        return [renorm*step for step in steps]
        
        
    def get_output_for(self, cost, updates=OrderedDict(), **kwargs):
        #memorizing steps for gradient clipping
        steps = []
        
        learning_rate = self.learning_rate
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon
        
        grads = theano.grad(cost, self.all_params)
        
        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)
        
        
        t = self.t_prev + one
        updates[self.t_prev] = t
        
        a_t = learning_rate*T.sqrt(one-beta2**t[0])/(one-beta1**t[0])
        for g_t, m_prev, v_prev in zip(grads, self.m_prevs, self.v_prevs):
            m_t = beta1*m_prev + (one-beta1)*g_t
            v_t = beta2*v_prev + (one-beta2)*g_t**2
            steps.append(a_t*m_t/(T.sqrt(v_t) + epsilon))

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            
        steps = self.gradient_clipping(steps)
        for param, step in zip(self.all_params, steps): updates[param] = param - step
        
        return cost
