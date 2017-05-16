import theano
import lasagne

class PaddingLayer(lasagne.layers.Layer):
    def __init__(self, incoming, left=None, right=None, both=None, val=0, **kwargs):
        super(PaddingLayer, self).__init__(incoming, **kwargs)
        
        if left is None and right is None:
            if both is None: both = [0 for i in range(len(self.input_shape))]
            left = right = both
            
        elif left is not None and right is None:
            right = [0 for i in range(len(self.input_shape))]
            
        elif left is None and right is not None:
            left = [0 for i in range(len(self.input_shape))]
            
        self.left=left
        self.right=right
        self.both=both
        self.val=val
        
    def get_output_shape_for(self, input_shape, **kwargs):
        output_shape = tuple()
        for s,l,r in zip(input_shape, self.left, self.right):
            output_shape += (None,) if s is None else (s + l + r,)
        return output_shape
    
    def get_output_for(self, input, **kwargs):
        sl, sr = tuple(), tuple([s for s in input.shape])
        for i, (l,r) in enumerate(zip(self.left, self.right)):
            tmp, sr = sr[0], sr[1:]
            if l > 0:
                pl = theano.tensor.zeros(sl + (l,) + sr)
                input = theano.tensor.concatenate([pl, input], axis=i)
            if r > 0:
                pr = theano.tensor.zeros(sl + (r,) + sr)
                input = theano.tensor.concatenate([input, pr], axis=i)
            sl = sl + (tmp + l + r,)
        return input