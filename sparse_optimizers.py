from keras.optimizers import SGD
from keras import backend as K
#from keras.utils.generic_utils import get_from_module
import numpy as np
from scipy.sparse import csr_matrix
import theano

# Sparse versions of Keras's built-in optimizers
# Based on Keras ver 1.0.6

class SparseSGD(SGD):
    '''Stochastic Gradient Descent optimizer that works with sparse data
    '''
    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]
        # momentum
        self.weights=[]
        for p in params:
            # csr_matrix( (3,4), dtype=int8 )
            if type(p.get_value()) is csr_matrix:
                m=csr_matrix(K.get_value(p).shape,dtype='float32')
                m=theano.shared(value=m, strict=False)
                self.weights.append(m)
            else:
                self.weights.append(K.variable(np.zeros(K.get_value(p).shape)))

        for p, g, m in zip(params, grads, self.weights):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

## aliases
#sparse_sgd = SparseSGD
#
#def get(identifier, kwargs=None):
#    return get_from_module(identifier, globals(), 'optimizer',
#                           instantiate=True, kwargs=kwargs)
