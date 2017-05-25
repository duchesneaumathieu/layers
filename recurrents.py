import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from loops import LoopInitialLayer
from specials import ZerosShapeLayer
from indexing import OutputSplitLayer


class LstmUnitLayer(lasagne.layers.MergeLayer):
    @staticmethod
    def get_hidd_and_cell_tap(x, num_inputs, debug=False):
        shape = (1, 'x', num_inputs)
        h = ZerosShapeLayer(x, shape)
        c = ZerosShapeLayer(x, shape)
        lh = LoopInitialLayer(h, debug=debug)
        lc = LoopInitialLayer(c, debug=debug)
        th = lh.tap(-1)
        tc = lc.tap(-1)
        return th, tc
    
    def __init__(self, incomings, Wh=lasagne.init.Normal(0.1),
        Wx=lasagne.init.Normal(0.1), **kwargs):
        #incomings are h,c,x
        super(LstmUnitLayer, self).__init__(incomings, **kwargs)
        
        self.num_units = self.input_shapes[0][1]
        assert self.num_units == self.input_shapes[1][1], 'hidden and cell must be the same shape'
        self.num_inputs = self.input_shapes[2][1]
        self.Wh = self.add_param(Wh, (self.num_units, 4*self.num_units), name='Wh')
        self.Wx = self.add_param(Wx, (self.num_inputs, 4*self.num_units), name='Wx')
        
        self.wci = self.add_param(lasagne.init.Normal(0.1), (1, self.num_units), name='wci')
        self.wcf = self.add_param(lasagne.init.Normal(0.1), (1, self.num_units), name='wcf')
        self.wco = self.add_param(lasagne.init.Normal(0.1), (1, self.num_units), name='wco')
        
        b_zeros = np.zeros((1, self.num_units), dtype=theano.config.floatX)
        b_ones = np.ones((1, self.num_units), dtype=theano.config.floatX)
        b = np.concatenate([b_zeros, b_ones, b_zeros, b_zeros], axis=1)
        self.b = self.add_param(lasagne.init.Constant(b), (1, 4*self.num_units), name='b')
        
    def slice_n(self, out, n):
        return out[:, n*self.num_units:(n+1)*self.num_units]
    
    def slice_all(self, out):
        return [self.slice_n(out, n) for n in range(4)]
    
    def get_output_for(self, inputs, **kwargs):
        h, c, x = inputs
        
        hWh = T.dot(h, self.Wh)
        xWx = T.dot(x, self.Wx)
        hxb = hWh + xWx + self.b
        hxbi, hxbf, hxbc, hxbo = self.slice_all(hxb)
        
        wi = c*self.wci
        wf = c*self.wcf
        
        i = T.nnet.sigmoid(hxbi + wi)
        f = T.nnet.sigmoid(hxbf + wf)
        c_ = f*c + i*T.tanh(hxbc)
        
        wo = c_*self.wco
        o = T.nnet.sigmoid(hxbo + wo)
        h_ = o*T.tanh(c_)
        
        return [h_, c_]
    
    def get_output_shape_for(self, shapes, **kwargs):
        return shapes
    
    def split(self):
        splits = []
        for n in range(2):
            splits += [OutputSplitLayer(self, idx=n)]
        return splits
    
class ConvLstmUnitLayer(lasagne.layers.MergeLayer):
    @staticmethod
    def get_hidd_and_cell_tap(x, num_inputs, debug=False):
        shape = (1, 'x', num_inputs, 'x', 'x')
        h = ZerosShapeLayer(x, shape)
        c = ZerosShapeLayer(x, shape)
        lh = LoopInitialLayer(h, debug=debug)
        lc = LoopInitialLayer(c, debug=debug)
        th = lh.tap(-1)
        tc = lc.tap(-1)
        return th, tc
    
    def __init__(self, incomings, 
        filter_size, Wh=lasagne.init.Normal(0.1),
        Wx=lasagne.init.Normal(0.1), **kwargs):
        
        #incomings are h,c,x
        super(ConvLstmUnitLayer, self).__init__(incomings, **kwargs)
        
        self.num_filters = self.input_shapes[0][1]
        assert self.num_filters == self.input_shapes[1][1], 'hidden and cell must be the same shape'
        self.num_inputs = self.input_shapes[2][1]
        self.filter_size = tuple(filter_size)
        
        self.Wh_shape = (4*self.num_filters, self.num_filters) + self.filter_size
        self.Wx_shape = (4*self.num_filters, self.num_inputs) + self.filter_size
        self.Wh = self.add_param(Wh, self.Wh_shape, name='Wh')
        self.Wx = self.add_param(Wx, self.Wx_shape, name='Wx')
        
        self.wc_shape = (1, self.num_filters, 1 ,1)
        self.wci = self.add_param(lasagne.init.Normal(0.1), self.wc_shape, name='wci')
        self.wcf = self.add_param(lasagne.init.Normal(0.1), self.wc_shape, name='wcf')
        self.wco = self.add_param(lasagne.init.Normal(0.1), self.wc_shape, name='wco')
        
        b_shape = (1, self.num_filters, 1, 1)
        b_zeros = np.zeros(b_shape, dtype=theano.config.floatX)
        b_ones = np.ones(b_shape, dtype=theano.config.floatX)
        b = np.concatenate([b_zeros, b_ones, b_zeros, b_zeros], axis=1)
        self.b = self.add_param(lasagne.init.Constant(b), b.shape, name='b')
        
    def slice_n(self, out, n):
        return out[:, n*self.num_filters:(n+1)*self.num_filters]
    
    def slice_all(self, out):
        return [self.slice_n(out, n) for n in range(4)]
    
    def get_output_for(self, inputs, **kwargs):
        h, c, x = inputs
        
        hWh = T.nnet.conv2d(h, self.Wh, self.input_shapes[0], self.Wh_shape,
            subsample=(1,1),
            border_mode='half',
            filter_flip=True)
        
        xWx = T.nnet.conv2d(x, self.Wx, self.input_shapes[2], self.Wx_shape,
            subsample=(1,1),
            border_mode='half',
            filter_flip=True)
        
        hxb = hWh + xWx + self.b
        hxbi, hxbf, hxbc, hxbo = self.slice_all(hxb)
        
        wi = c*self.wci
        wf = c*self.wcf
        
        i = T.nnet.sigmoid(hxbi + wi)
        f = T.nnet.sigmoid(hxbf + wf)
        c_ = f*c + i*T.tanh(hxbc)
        
        wo = c_*self.wco
        o = T.nnet.sigmoid(hxbo + wo)
        h_ = o*T.tanh(c_)
        
        return [h_, c_]
    
    def get_output_shape_for(self, shapes, **kwargs):
        return shapes[:-1]
    
    def split(self):
        splits = []
        for n in range(2):
            splits += [OutputSplitLayer(self, idx=n)]
        return splits

class BiGRULayer(lasagne.layers.Layer):
    def __init__(
        self, incoming, forward_num_units,
        backward_num_units=None, share=False,
        concatenate=False, stack=False,
        output_form='sum', learn_init=False,
        grad_clipping=5, **kwargs):
        
        if output_form not in ['sum', 'concatenate', 'top']:
            raise ValueError('BiGRULayer: output most be sum, concatenate or top')
            
        if output_form == 'top' and not stack:
            raise ValueError(('BiGRULayer: It doesn\'t make sense not'
                              ' to stack and output only '
                              'the top GRU (first GRU is useless)'))
        
        if output_form == 'sum' and backward_num_units is not None and forward_num_units != backward_num_units:
            raise ValueError(('BiGRULayer: If we sum, forward_num_units most be'
                              ' equal to backward_num_units'))
            
        if stack and share and incoming.output_shape[2] != forward_num_units:
            raise ValueError(('BiGRULayer: If we stack and share parameters'
                              ' we need the first GRU to keep '
                              'the shape unchanged'))
            
        if share and backward_num_units is not None and forward_num_units != backward_num_units:
            raise ValueError(('BiGRULayer: If we stack and share parameters'
                              ' backward_num_units most be equal to forward_num_units'))
        
        
        super(BiGRULayer, self).__init__(incoming, **kwargs)
        
        self.forward_num_units = forward_num_units
        self.backward_num_units = forward_num_units if backward_num_units is None else backward_num_units
        self.share = share
        self.concatenate = concatenate
        self.stack = stack
        self.output_form = output_form
        self.learn_init = learn_init
        self.grad_clipping = grad_clipping
        
        self.forward = lasagne.layers.GRULayer(
            incoming, self.forward_num_units,
            resetgate=lasagne.layers.Gate(W_cell=None),
            updategate=lasagne.layers.Gate(W_cell=None),
            hidden_update=lasagne.layers.Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
            hid_init=lasagne.init.Constant(0.),
            backwards=False, learn_init=learn_init, 
            grad_clipping=grad_clipping)
        
        backward_in = self.forward if stack else incoming
        params = self.forward.get_params()
        
        W_in_to_updategate = params[0] if share else lasagne.init.Normal(0.1)
        W_hid_to_updategate = params[1] if share else lasagne.init.Normal(0.1)
        b_updategate = params[2] if share else lasagne.init.Constant(0.)
        
        W_in_to_resetgate = params[3] if share else lasagne.init.Normal(0.1)
        W_hid_to_resetgate = params[4] if share else lasagne.init.Normal(0.1)
        b_resetgate = params[5] if share else lasagne.init.Constant(0.)
        
        W_in_to_hidden_update = params[6] if share else lasagne.init.Normal(0.1)
        W_hid_to_hidden_update = params[7] if share else lasagne.init.Normal(0.1)
        b_hidden_update = params[8] if share else lasagne.init.Constant(0.)
        
        hid_init = params[9] if share else lasagne.init.Constant(0.)
        
        self.backward = lasagne.layers.GRULayer(
            backward_in, self.backward_num_units,
            resetgate=lasagne.layers.Gate(
                W_in=W_in_to_resetgate,
                W_hid=W_hid_to_resetgate,
                b=b_resetgate,
                W_cell=None),
            updategate=lasagne.layers.Gate(
                W_in=W_in_to_updategate,
                W_hid=W_hid_to_updategate,
                b=b_updategate,
                W_cell=None),
            hidden_update=lasagne.layers.Gate(
                W_in=W_in_to_hidden_update,
                W_hid=W_hid_to_hidden_update,
                b=b_hidden_update,
                W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
            hid_init=hid_init,
            backwards=True, learn_init=learn_init, 
            grad_clipping=grad_clipping)
        
        if output_form == 'sum':
            self.output = ElemwiseSumLayer([self.forward, self.backward])
        elif output_form == 'concatenate':
            self.output = ConcatLayer([self.forward, self.backward], axis=2)
        elif output_form == 'top':
            self.output = self.backward
            
        # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        self.input_layer = self.output

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var

    def get_output_shape_for(self, input_shape):
        if self.output_form == 'top': s2 = self.backward_num_units
        elif self.output_form == 'sum': s2 = self.forward_num_units
        elif self.output_form == 'concatenate': s2 = self.backward_num_units + self.forward_num_units
        return input_shape[:2] + (s2,)
    
class MapsToRNN(lasagne.layers.Layer):
    def __init__(self, incoming, patch_shape, direction='horizontal', **kwargs):
        super(MapsToRNN, self).__init__(incoming, **kwargs)
        self.patch_shape = patch_shape
        assert direction in ['horizontal', 'vertical'], 'direction most be horizontal or vertical'
        self.direction = direction
        input = get_output(incoming)
        (ph, pw) = self.patch_shape
        shape = list(input.shape)
        for i in range(len(shape)):
            if incoming.output_shape[i] is not None:
                shape[i] = incoming.output_shape[i]
        (bs, nc, ch, cw) = shape
        nph = ch // ph
        npw = cw // pw
        self.all_dim = (bs, nc, nph, ph, npw, pw)
        
    def get_output_for(self, input, **kwargs):
        (bs, nc, nph, ph, npw, pw) = self.all_dim
        tmp = input.reshape(self.all_dim)
        if self.direction == 'horizontal':
            tmp = tmp.transpose((0, 2, 4, 3, 5, 1))
            output = tmp.reshape((-1, npw, ph*pw*nc))
        else:
            tmp = tmp.transpose((0, 4, 2, 3, 5, 1))
            output = tmp.reshape((-1, nph, ph*pw*nc))
        return output
        
    def get_output_shape_for(self, input_shape):
        # bs*#H, #W, ph * pw * cc
        ish = list(input_shape)
        (ph, pw) = self.patch_shape
        if self.direction == 'vertical':
            ish[2], ish[3] = ish[3], ish[2]
            ph, pw = pw, ph
        
        if ish[0] is None or ish[2] is None:
            s0 = None
        else:
            s0 = ish[0] * (ish[2] // ph)
            
        if ish[3] is None:
            s1 = None
        else:
            s1 = ish[3] // pw
        
        if ish[1] is None:
            s2 = None
        else:
            s2 = ph * pw * ish[1]
            
        return (s0, s1, s2)
    

class RNNToMaps(lasagne.layers.Layer): #vertical to maps
    def __init__(
        self, incoming,
        all_dim, direction='horizontal',
        **kwargs):
        
        super(RNNToMaps, self).__init__(incoming, **kwargs)
        
        assert direction in ['horizontal', 'vertical'], 'direction most be horizontal or vertical'
        
        self.all_dim=all_dim
        self.direction=direction
        
    def get_output_for(self, input, **kwargs):
        (bs, nc, nph, ph, npw, pw) = self.all_dim
        
        if self.direction == 'horizontal':
            (_, npw, birnn_hidden) = input.shape
            tmp = input.reshape((bs, nph, npw, birnn_hidden))
            tmp = tmp.transpose((0,3,1,2))
            
        else:
            (_, nph, birnn_hidden) = input.shape
            tmp = input.reshape((bs, npw, nph, birnn_hidden))
            tmp = tmp.transpose((0,3,2,1))
            
        output = tmp
        return output
        
    def get_output_shape_for(self, input_shape):
        # bs, birnn_hidden, #H, #W
        
        s0 = self.all_dim[0] if isinstance(self.all_dim[0], int) else None
        
        s1 = self.input_layer.output_shape[2]
        
        s2 = self.all_dim[2] if isinstance(self.all_dim[2], int) else None
        
        s3 = self.all_dim[4] if isinstance(self.all_dim[4], int) else None
            
        return (s0, s1, s2, s3)
    
class ResReNetLayer(lasagne.layers.Layer):
    def __init__(
        self, incoming,
        num_maps,
        
        horizontal_num_units=None,
        horizontal_share=True,
        horizontal_stack=False,
        horizontal_output_form='sum',
        
        vertical_num_units=None,
        vertical_share=False,
        vertical_stack=True,
        vertical_output_form='top',
        
        residual=True,
        patch_size=(1,1),
        bottleneck=None,
        batch_norm=True,
        last_batch_norm=False,
        **kwargs):
        
        super(ResReNetLayer, self).__init__(incoming, **kwargs)
        
        if horizontal_num_units is None:
            if bottleneck is None: horizontal_num_units = num_maps
            else: horizontal_num_units = bottleneck
        if vertical_num_units is None: vertical_num_units = horizontal_num_units
            
        self.num_maps=num_maps
            
        self.horizontal_num_units=horizontal_num_units
        self.horizontal_share=horizontal_share
        self.horizontal_stack=horizontal_stack
        self.horizontal_output_form=horizontal_output_form
        
        self.vertical_num_units=vertical_num_units
        self.vertical_share=vertical_share
        self.vertical_stack=vertical_stack
        self.vertical_output_form=vertical_output_form
        
        self.residual=residual
        self.patch_size=patch_size
        self.bottleneck=bottleneck
        self.batch_norm=batch_norm
        self.last_batch_norm=last_batch_norm
        
        if bottleneck is None:
            horizontal_input = incoming
        else:
            horizontal_input = Conv2DLayer(incoming, bottleneck, (1,1), **kwargs)
            if batch_norm: horizontal_input = lasagne.layers.batch_norm(horizontal_input)
                
        horizontal_input = MapsToRNN(horizontal_input, self.patch_size, direction='horizontal')
        
        horizontal_output = BiGRULayer(
            horizontal_input,
            horizontal_num_units,
            share=horizontal_share,
            stack=horizontal_stack,
            output_form=horizontal_output_form,
            **kwargs)
        
        horizontal_output = RNNToMaps(horizontal_output, horizontal_input.all_dim, direction='horizontal')
        
        vertical_input = horizontal_input = Conv2DLayer(
            horizontal_output, horizontal_num_units, (1,1), **kwargs)
        
        if batch_norm: vertical_input = lasagne.layers.batch_norm(vertical_input)
        
        vertical_input = MapsToRNN(vertical_input, self.patch_size, direction='vertical')
        
        vertical_output = BiGRULayer(
            vertical_input,
            vertical_num_units,
            share=vertical_share,
            stack=vertical_stack,
            output_form=vertical_output_form,
            **kwargs)
        
        vertical_output = RNNToMaps(vertical_output, vertical_input.all_dim, direction='vertical')
        
        if residual:
            core = Conv2DLayer(incoming, self.num_maps, (1,1), **kwargs)
            residue = Conv2DLayer(vertical_output, self.num_maps, (1,1), **kwargs)
            output = ElemwiseSumLayer([core, residue])
            output = Conv2DLayer(output, self.num_maps, (1,1), **kwargs)
            
        else:
            output = Conv2DLayer(vertical_output, self.num_maps, (1,1), **kwargs)
            
        if last_batch_norm: output = lasagne.layers.batch_norm(output)
            
        self.output = output
            
        # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        self.input_layer = self.output
        

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var
    
    def get_output_shape_for(self, input_shape):
        return input_shape[:1] + (self.num_maps,) + input_shape[2:]