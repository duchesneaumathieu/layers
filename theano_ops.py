import numpy as np
import theano
import theano.tensor as T

from munkres import Munkres


#Batch dot product
def batch_dot(A, B):
    C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])
    return C.sum(axis=-2)


class PermutationOp(theano.Op):
    __props__ = ()

    def make_node(self, x, y):
        x = T.as_tensor_variable(x)
        y = T.as_tensor_variable(y)
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x, y], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        y = inputs[1]
        t = list()
        for i in range(x.shape[0]):
            t.append(x[i][y[i]])
        z = output_storage[0]
        z[0] = np.array(t)

    def infer_shape(self, node, i0_shapes):
        return [i0_shapes[0]]

    def grad(self, inputs, output_grads):
        return [theano.gradient.disconnected_grad(inputs[0]),
                theano.gradient.disconnected_grad(inputs[1].astype('float32'))]

permutationOp = PermutationOp()

class MunkresOp(theano.Op):
    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x], [T.imatrix()])

    def perform(self, node, inputs, output_storage):
        #inputs: a.shape=(bs,sq,sq)
        #return the permutation of the second dimension (axis=2)
        a = np.array(inputs[0]) #copy
        m = Munkres()
        l = []
        for i in range(a.shape[0]):
            l.append([j for i,j in m.compute(a[i])])
        z = output_storage[0]
        z[0] = np.array(l, dtype=np.int32)

    def infer_shape(self, node, i0_shapes):
        return [i0_shapes[0][:2]]

    def grad(self, inputs, output_grads):
        return [theano.gradient.disconnected_grad(inputs[0])]
    
munkresOp = MunkresOp()

class AssignmentMaskOp(theano.Op):
    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        return theano.Apply(self, [x], [T.ftensor3()])

    def perform(self, node, inputs, output_storage):
        #inputs: a.shape=(bs,sq)
        a = inputs[0]
        bs, sq = a.shape
        r = np.zeros((bs, sq, sq), dtype=np.float32)
        for i in range(bs):
            for j in range(sq):
                r[i, j, a[i, j]] = 1
        z = output_storage[0]
        z[0] = r

    def infer_shape(self, node, i0_shapes):
        bs, sq = i0_shapes[0]
        return [(bs, sq, sq)]

    def grad(self, inputs, output_grads):
        return [theano.gradient.disconnected_grad(inputs[0].astype('float32'))]
    
assignmentMaskOp = AssignmentMaskOp()

class Unravel_index_Op(theano.Op):
    __props__ = ()

    def make_node(self, x, y):
        x = T.as_tensor_variable(x)
        y = T.as_tensor_variable(y)
        return theano.Apply(self, [x, y], [T.lmatrix().type()])

    def perform(self, node, inputs, output_storage):
        z = output_storage[0]
        z[0] = np.stack(np.unravel_index(inputs[0], inputs[1]))

    def infer_shape(self, node, i0_shapes):
        return [[i0_shapes[1][0],i0_shapes[0][0]]]

    def grad(self, inputs, output_grads):
        return [theano.gradient.disconnected_grad(inputs[0]),
                theano.gradient.disconnected_grad(inputs[1].astype('float32'))]

unravel_index_Op = Unravel_index_Op()