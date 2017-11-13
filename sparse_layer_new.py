import numpy as np
from keras.layers import Dense
from keras.initializers import Initializer
from keras import backend as K

class AdjacencyInitializer(Initializer):
    def __init__(self, adjacency_mat=1):
        # Default value is 1 which translates to a dense (fully connected) layer
        self.adjacency_mat = adjacency_mat

    def __call__(self, shape, dtype=None):
        return K.constant(self.adjacency_mat, shape=shape, dtype=dtype)

    def get_config(self):
        return {'adjacency_mat':self.adjacency_mat}


class Sparse(Dense):
    def __init__(self,
                 adjacency_mat=None, #Specifies which inputs (rows) are connected to which outputs (columns)
                 *args,
                 **kwargs):
        self.adjacency_mat = adjacency_mat
        self.adjacency_tensor = K.variable(value=adjacency_mat)
        if adjacency_mat is not None:
            units = adjacency_mat.shape[1]
        super().__init__(units=units, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        # Ensure we set weights to zero according to adjancency matrix
        self.adjacency_tensor = self.add_weight(shape=(input_dim, self.units),
                                                initializer=AdjacencyInitializer(self.adjacency_mat),
                                                name='adjacency_matrix',
                                                trainable=False)
        self.kernel = self.kernel * self.adjacency_tensor

    def call(self, inputs):
        output = self.kernel * self.adjacency_tensor
        output = K.dot(inputs, output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
        
    def get_config(self):
        config = {
            'adjacency_mat': self.adjacency_mat.tolist()
        }
        base_config = super().get_config()
        base_config.pop('units', None)
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        adjacency_mat_as_list = config['adjacency_mat']
        config['adjacency_mat'] = np.array(adjacency_mat_as_list)
        return cls(**config)
