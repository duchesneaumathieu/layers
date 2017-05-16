import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *


#DRAW https://arxiv.org/pdf/1502.04623v2.pdf
#from theano_ops import batch_dot
class GaussianFiltersLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, keepaspect=False, tol=1e-7, **kwargs):
        super(GaussianFiltersLayer, self).__init__(incomings, **kwargs)
        self.keepaspect=keepaspect
        self.tol=tol
    
    def get_output_for(self, inputs, **kwargs):
        floatX = theano.config.floatX
        
        bbox, in_shape, out_shape = inputs
        
        crx, cry, lrx, lry, scp = bbox[:,0], bbox[:,1], T.exp(bbox[:,2]), T.exp(bbox[:,3]), T.exp(bbox[:,4])
        B, A = in_shape[0], in_shape[1]
        M, N = out_shape[0], out_shape[1]
        
        Af = A.astype(floatX)
        Bf = B.astype(floatX)
        Nf = N.astype(floatX)
        Mf = M.astype(floatX)
        
        gx = (Af-1)*(crx+1)/2.
        gy = (Bf-1)*(cry+1)/2.
        
        dx = (Af-1)/(Nf-1)*lrx
        dy = (Bf-1)/(Mf-1)*lry
        
        if self.keepaspect:
            d = T.maximum(dx, dy)
            dx, dy = d, d
        
        sx = scp*dx/2.
        sy = scp*dy/2.
        
        rngx = T.arange(N, dtype=floatX)-Nf/2.+0.5
        rngy = T.arange(M, dtype=floatX)-Mf/2.+0.5
        
        mux = gx.dimshuffle([0,'x']) + dx.dimshuffle([0,'x'])*rngx
        muy = gy.dimshuffle([0,'x']) + dy.dimshuffle([0,'x'])*rngy
        
        a = T.arange(A, dtype=floatX)
        b = T.arange(B, dtype=floatX)
        
        Fx = T.exp(-(a - mux.dimshuffle([0,1,'x']))**2 / 2. / sx.dimshuffle([0,'x','x'])**2 )
        Fy = T.exp(-(b - muy.dimshuffle([0,1,'x']))**2 / 2. / sy.dimshuffle([0,'x','x'])**2 )
        Fx = Fx / (Fx.sum(axis=-1).dimshuffle(0, 1, 'x') + self.tol)
        Fy = Fy / (Fy.sum(axis=-1).dimshuffle(0, 1, 'x') + self.tol)
        
        return Fx, Fy
        
    def get_output_shape_for(self, shapes, **kwargs):
        (h,w), (h_,w_) = shapes[1:]
        return [(w, w_), (h, h_)]

class GaussianReadLayer(lasagne.layers.MergeLayer):
    def get_output_for(self, inputs, **kwargs):
        input, filters = inputs
        Fx, Fy = filters
        bs, fs, h, w = input.shape
        input = input.reshape((bs*fs, h, w))
        w_ = Fx.shape[1]
        h_ = Fy.shape[1]
        Fx = T.repeat(Fx, fs, axis=0)
        Fy = T.repeat(Fy, fs, axis=0)
        
        output = T.batched_dot(Fy, T.batched_dot(input, Fx.transpose(0,2,1)))
        return output.reshape((bs, fs, h_, w_))
    
    def get_output_shape_for(self, shapes, **kwargs):
        (bs, fs, _, _), ((_, w), (_, h)) = shapes
        return (bs, fs, h, w)
    
class GaussianWriteLayer(lasagne.layers.MergeLayer):
    def get_output_for(self, inputs, **kwargs):
        input, filters = inputs
        Fx, Fy = filters
        bs, fs, h, w = input.shape
        input = input.reshape((bs*fs, h, w))
        w_ = Fx.shape[2]
        h_ = Fy.shape[2]
        Fx = T.repeat(Fx, fs, axis=0)
        Fy = T.repeat(Fy, fs, axis=0)
        
        output = T.batched_dot(Fy.transpose(0,2,1), T.batched_dot(input, Fx))
        return output.reshape((bs, fs, h_, w_))
    
    def get_output_shape_for(self, shapes, **kwargs):
        (bs, fs, _, _), ((w, _), (h, _)) = shapes
        return (bs, fs, h, w)