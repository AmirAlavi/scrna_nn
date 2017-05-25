from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.merge import Concatenate

from bio_sparse_layer import BioSparseLayer
from util import ScrnaException

autoencoder_model_names = ['1layer_ae']
ppitf_model_names = ['2layer_ppitf']

def get_1layer_autoencoder(hidden_layer_size, input_dim, activation_fcn='tanh'):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=input_dim, activation=activation_fcn))
    model.add(Dense(input_dim, activation=activation_fcn))
    return model

def get_2layer_ppitf(second_hidden_layer_size, input_dim, ppitf_groups_mat, output_dim, activation_fcn='tanh'):
    left_branch = Sequential()
    left_branch.add(BioSparseLayer(input_dim=input_dim, activation=activation_fcn, input_output_mat=ppitf_groups_mat.transpose()))
    right_branch = Sequential()
    right_branch.add(Dense(100, input_dim=input_dim))
    merged = Concatenate([left_branch, right_branch])
    print("Type of merged: ", type(merged))
    model = Sequential()
    model.add(merged)
    #model = Concatenate([left_branch, right_branch])
    model.add(Dense(second_hidden_layer_size, activation=activation_fcn))
    model.add(Dense(output_dim, activation='softmax'))

def get_nn_model(model_name, hidden_layer_sizes, input_dim, activation_fcn='tanh', ppitf_groups_mat=None, output_dim=None):
    print(hidden_layer_sizes)
    if model_name == '1layer_ae':
        return get_1layer_autoencoder(hidden_layer_sizes[0], input_dim, activation_fcn)
    elif model_name == '2layer_ppitf':
        return get_2layer_ppitf(hidden_layer_sizes[0], input_dim, ppitf_groups_mat, output_dim, activation_fcn)
    else:
        raise ScrnaException("Bad neural network name: " + model_name)
