from keras.models import Sequential
from keras.layers.core import Dense, Flatten

from bio_sparse_layer import BioSparseLayer
from util import ScrnaException

autoencoder_model_names = ['1layer_ae']

def get_1layer_autoencoder(hidden_layer_size, input_dim, activation_fcn='tanh'):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=input_dim, activation=activation_fcn))
    model.add(Dense(input_dim, activation=activation_fcn))
    return model

def get_2layer_ppitf(second_hidden_layer_size, input_dim, activation_fcn='tanh'):
    left_branch = Sequential()
    #left_branch.add(BioSparseLayer(input_dim=input_dim, activation=activation_fcn, input_output_mat=))

def get_nn_model(model_name, hidden_layer_sizes, input_dim, activation_fcn='tanh'):
    print(hidden_layer_sizes)
    if model_name == '1layer_ae':
        return get_1layer_autoencoder(hidden_layer_sizes[0], input_dim, activation_fcn)
    else:
        raise ScrnaException("Bad neural network name: " + model_name)
