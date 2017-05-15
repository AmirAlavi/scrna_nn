from keras.models import Sequential
from keras.layers.core import Dense

from util import ScrnaException


def get_1layer_autoencoder(hidden_layer_size, input_dim, activation_fcn='tanh'):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=input_dim, activation=activation_fcn))
    model.add(Dense(input_dim, activation=activation_fcn))
    return model


def get_nn_model(model_name, hidden_layer_sizes, input_dim, activation_fcn='tanh'):
    print(hidden_layer_sizes)
    if model_name == '1layer_ae':
        return get_1layer_autoencoder(hidden_layer_sizes[0], input_dim, activation_fcn)
    else:
        raise ScrnaException("Bad neural network name: " + model_name)
