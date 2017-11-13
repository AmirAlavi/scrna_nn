from keras.optimizers import SGD, RMSprop
from keras import backend as K
#from keras.legacy import interfaces
#from keras.utils.generic_utils import get_from_module
import numpy as np
from scipy.sparse import csr_matrix
import theano
from theano import sparse

# Sparse versions of Keras's built-in optimizers
# Based on Keras ver 1.0.6

class SparseSGD(SGD):
    '''Stochastic Gradient Descent optimizer that works with sparse data
    '''
    def get_updates(self, params, constraints, loss):
        """From Chieh Lin's code for:

        Chieh Lin, Siddhartha Jain, Hannah Kim, Ziv Bar-Joseph;
        Using neural networks for reducing the dimensions of single-cell RNA-Seq data,
        Nucleic Acids Research,
        Volume 45, Issue 17, 29 September 2017, Pages e156,
        https://doi.org/10.1093/nar/gkx681
        """
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

class SparseRMSprop(RMSprop):
    def get_updates(self, params, constraints, loss):
        print("RMS loss: ", loss)
        grads = self.get_gradients(loss, params)
        accumulators = []
        for p in params:
            if type(p.get_value()) is csr_matrix:
                m = csr_matrix(K.get_value(p).shape, dtype='float32')
                m = theano.shared(value=m, strict=False)
                accumulators.append(m)
            else:
                accumulators.append(K.zeros(K.int_shape(p), dtype=K.dtype(p)))
        # shapes = [K.get_variable_shape(p) for p in params]
        # accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            print("g.type: ", g.type)
            if "Sparse" in str(g.type): # HACK
                new_a = self.rho * a + (1. - self.rho) * sparse.sqr(g)
            else:
                new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            if "Sparse" in str(new_a.type):
                #new_p = p - lr * g / (sparse.sqrt(new_a) + (self.epsilon * sparse.sp_ones_like(new_a)))
                #new_p = p - lr * g * sparse.structured_pow(K.sqrt(sparse.dense_from_sparse(new_a)) + self.epsilon, -1)
                new_p = p - lr * g / (sparse.sqrt(new_a) + self.epsilon)
            else:
                new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    
    # def get_updates(self, params, constraints, params):
    #     grads = self.get_gradients(loss, params)
    #     accumulators = []
    #     for p in params:
    #         if type(p.get_value()) is csr_matrix:
    #             m = csr_matrix(K.get_value(p).shape, dtype='float32')
    #             m = theano.shared(value=m, strict=False)
    #             accumulators.append(m)
    #         else:
    #             accumulators.append(K.zeros(K.int_shape(p), dtype=K.dtype(p)))
    #     self.weights = accumulators
    #     self.updates = [K.update_add(self.iterations, 1)]
    #     lr = self.lr
    #     if self.initial_decay > 0:
    #         lr *= (1. / (1. + self.decay * K.cast(self.iterations,
    #                                               K.dtype(self.decay))))

    #     for p, g, a in zip(params, grads, accumulators):
    #         # update accumulator
    #         new_a = self.rho * a + (1. - self.rho) * K.square(g)
    #         self.updates.append(K.update(a, new_a))
    #         new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

    #         # Apply constraints.
    #         if getattr(p, 'constraint', None) is not None:
    #             new_p = p.constraint(new_p)

    #         self.updates.append(K.update(p, new_p))
    #     return self.updates

## aliases
#sparse_sgd = SparseSGD
#
#def get(identifier, kwargs=None):
#    return get_from_module(identifier, globals(), 'optimizer',
#                           instantiate=True, kwargs=kwargs)
