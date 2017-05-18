import numpy as np

from scipy.sparse import csr_matrix
from keras.layers import Dense
from keras import backend as K
from keras.engine import InputSpec
from theano import sparse
import theano

# Hack: Keras 2 does not have a get_fans function in the initializations.py
# module. This used to be in the initilization.py module in Keras 1. Copying
# it here for now. TODO: move to Keras 2 API
def get_fans(shape, dim_ordering='th'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif dim_ordering == 'tf':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

class BioSparseLayer(Dense):
    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 input_output_mat=None,
                 group_gene_dict=None,
                 bias=True, input_dim=None, **kwargs):
        if not input_output_mat:
            raise ValueError("Must provide input_output_mat to BioSparseLayer constructor!")
        self.input_output_mat=input_output_mat
        self.group_gene_dict=group_gene_dict
        output_dim = self.input_output_mat.shape[1]
        super(BioSparseLayer, self).__init__(output_dim, init, activation, weights, W_regularizer, b_regularizer, activity_regularizer, W_constraint, b_constraint, bias, input_dim, **kwargs)

    def build_helper(self, input_shape, W):
        """This function contains the logic taken directly from Keras' Dense, placed here to make the difference
        between BioSparseLayer.build and Dense.build more clear. In other words, build is now a "template method".
        """
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        print("Input dimension: ", input_dim)
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        # The difference between built-in Dense and BioSparseLayer
        self.W = W

        if self.bias:
            self.b = K.zeros((self.output_dim,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def build(self, input_shape):
        assert len(input_shape) == 2

        temp_W = np.asarray(self.input_output_mat, dtype=K.floatx())
        if self.input_output_mat is not None:
            fan_in, fan_out = get_fans((input_shape[1], self.output_dim), dim_ordering='th')
            print("Fan in, Fan out:")
            print (fan_in, fan_out)
            scale = np.sqrt(6. / (fan_in + fan_out))
            for i in range(self.input_output_mat.shape[0]):
                for j in range(self.input_output_mat.shape[1]):
                    if  self.input_output_mat[i,j] == 1.:
                        temp_W[i,j]=np.random.uniform(low=-scale, high=scale)

        temp_W=csr_matrix(temp_W)
        W=theano.shared(value=temp_W, name='{}_W'.format(self.name), strict=False)

        self.build_helper(input_shape, W)

    def call(self, x, mask=None):
        output = sparse.structured_dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

    def set_weights(self, weights):
        '''Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).
        '''
        params = self.trainable_weights + self.non_trainable_weights
        if len(params) != len(weights):
            raise Exception('You called `set_weights(weights)` on layer "' + self.name +
                            '" with a  weight list of length ' + str(len(weights)) +
                            ', but the layer was expecting ' + str(len(params)) +
                            ' weights. Provided weights: ' + str(weights))
        if not params:
            return
        weight_value_tuples = []
        for  p, w in zip(params, weights):
            weight_value_tuples.append((p, w))
        weight_value_tuples
        for x, value in weight_value_tuples:
            x.set_value(value)
