import numpy as np

from scipy.sparse import csr_matrix
from keras.layers import Dense
from keras import backend as K
from keras.engine import InputSpec
#from keras.legacy import interfaces
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
    def __init__(self, units=0,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_output_mat=None,
                 group_gene_dict=None,
                 **kwargs):
        # if input_output_mat == None:
        #     raise ValueError("Must provide input_output_mat to BioSparseLayer constructor!")
        self.input_output_mat=input_output_mat
        self.group_gene_dict=group_gene_dict
        # Hack, necessary because of the way deserialization from json calls this constructor,
        # doesn't provide the input_output_mat, though it doesn't matter because when we load
        # a model from a file, we will eventually load weights we have already trained.
        if self.input_output_mat is not None:
            units = self.input_output_mat.shape[1]
        super().__init__(units=units, kernel_initializer=kernel_initializer, activation=activation, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, use_bias=use_bias, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        print("Input dimension: ", input_dim)


        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        # The difference between built-in Dense and BioSparseLayer
        self.kernel = self.get_kernel(input_shape)
        print("kernel type:", type(self.kernel))

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def get_kernel(self, input_shape):
        temp_W = np.asarray(self.input_output_mat, dtype=K.floatx())
        if self.input_output_mat is not None:
            fan_in, fan_out = get_fans((input_shape[1], self.units), dim_ordering='th')
            print("Fan in, Fan out:")
            print (fan_in, fan_out)
            scale = np.sqrt(6. / (fan_in + fan_out))
            for i in range(self.input_output_mat.shape[0]):
                for j in range(self.input_output_mat.shape[1]):
                    if  self.input_output_mat[i,j] == 1.:
                        temp_W[i,j]=np.random.uniform(low=-scale, high=scale)

        temp_W=csr_matrix(temp_W)
        W=theano.shared(value=temp_W, name='{}_W'.format(self.name), strict=False)
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(W))
        if self.kernel_constraint is not None:
            self.constraints[W] = self.kernel_constraint
        self._trainable_weights.append(W)

        return W

    def call(self, inputs):
        print("kernel type: ", type(self.kernel))
        print("bias type: ", type(self.bias))
        print("input type: ", type(inputs))
        output = sparse.structured_dot(inputs, self.kernel)
        if self.use_bias:
            print(self.kernel.shape)
            print(inputs.shape)
            print(self.bias.shape)
            output += self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def set_weights(self, weights):
        '''Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).
        '''
        print("you used to call me on my cell phone")
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
