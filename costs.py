import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *

class PixelClassificationLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        (bs, cc, h, w) = input.shape
        tmp = input.transpose((0,2,3,1))
        tmp = tmp.reshape((bs*h*w, cc))
        tmp = T.nnet.nnet.softmax(tmp)
        output = tmp.reshape((bs, h, w, cc))
        return output
    
    def get_output_shape_for(self, shape):
        return (shape[0], shape[2], shape[3], shape[1])
    
class StableSoftMaxCrossEntropyLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings,  axis=1, **kwargs):
        super(StableSoftMaxCrossEntropyLayer, self).__init__(incomings, **kwargs)
        self.axis = axis
    
    def get_output_for(self, inputs, **kwargs):
        input, target = inputs
        in_dev = input - input.max(axis=self.axis, keepdims=True)
        log_softmax = in_dev - T.log(T.sum(T.exp(in_dev), axis=self.axis, keepdims=True))
        cross_entropy = -T.sum(target * log_softmax, axis=self.axis)
        return T.mean(cross_entropy)

    def get_output_shape_for(self, input_shape):
        return ()

class StableBinaryCrossEntropyLayer(lasagne.layers.Layer):
    def __init__(self, incoming, target, **kwargs):
        super(StableBinaryCrossEntropyLayer, self).__init__(incoming, **kwargs)
        self.target = get_output(target)

    def get_output_for(self, input, **kwargs):
        small = T.log(1+T.exp(input)) - input*self.target
        big = (1-self.target)*input + T.log(1+T.exp(-input))
        stable = T.switch(T.lt(input, 0), small, big)
        return stable#T.mean(stable)

    def get_output_shape_for(self, input_shape):
        return (1,)
    
class SoftIoULayer(lasagne.layers.MergeLayer):
    def get_output_for(self, inputs, **kwargs):
        #inputs: dt.shape=(bs, sq, h, w), gt.shape=(bs, sq, h, w)
        #return: shape=()
        dt, gt = inputs
        tmp = (dt * gt).sum(axis=(2,3))
        return (tmp/((dt + gt).sum(axis=(2,3)) - tmp)).mean()
    
    def get_output_shape_for(self, shapes, **kwargs):
        return ()
    
class MultipleSoftIoULayer(lasagne.layers.MergeLayer):
    #SoftIoU like in https://arxiv.org/pdf/1605.09410.pdf
    def __init__(self, incomings, **kwargs):
        if len(incomings) != 2:
            raise ValueError('We need [dt, gt] as input')
            
        super(MultipleSoftIoULayer, self).__init__(incomings, **kwargs)
        
    def get_output_for(self, inputs, **kwargs):
        #inputs: dt.shape=(bs, sq1, h, w), gt.shape=(bs,sq2, h, w)
        #return: shape=(bs, sq1, sq2)
        dt, gt = inputs
        dt = dt.dimshuffle([0,1,'x',2,3])
        gt = gt.dimshuffle([0,'x',1,2,3])
        tmp = (dt * gt).sum(axis=(3,4))
        return tmp/((dt + gt).sum(axis=(3,4)) - tmp)
    
    def get_output_shape_for(self, shapes, **kwargs):
        dts, gts = shapes
        s0 = dts[0]
        s1 = dts[1]
        s2 = gts[1]
        return (s0, s1, s2)