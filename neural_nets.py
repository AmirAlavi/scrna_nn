# import pdb; pdb.set_trace()
import pickle
import math

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Lambda
from keras import backend as K

from bio_sparse_layer import BioSparseLayer
from util import ScrnaException


def load_model_weights_from_pickle(model, path):
    with open(path, 'rb') as fp:
        weight_list = pickle.load(fp)
    for layer, weights in zip(model.layers, weight_list):
        layer.set_weights(weights)

def save_model_weights_to_pickle(model, path):
    weight_list = [layer.get_weights() for layer in model.layers]
    with open(path, 'wb') as fp:
        pickle.dump(weight_list, fp)

def save_trained_nn(model, architecture_path, weights_path):
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)
    save_model_weights_to_pickle(model, weights_path)

def load_trained_nn(architecture_path, weights_path):
    custom_obj = {'BioSparseLayer': BioSparseLayer}
    model = model_from_json(open(architecture_path).read(), custom_objects=custom_obj)
    print(model.summary())
    load_model_weights_from_pickle(model, weights_path)
    # Must compile to use it for evaluatoin, but not going to train
    # this model, so we can compile it arbitrarily
    model.compile(optimizer='sgd', loss='mse')
    return model

def get_pretrained_weights(pt_file):
    weight_list = []
    with open(pt_file, 'rb') as weights_file:
        weight_list = pickle.load(weights_file)
    return weight_list

def set_pretrained_weights(model, pt_file):
    weight_list = get_pretrained_weights(pt_file)
    if len(model.layers) != len(weight_list):
        raise ScrnaException("Pretrained model weights do not match this architecture!")
    for layer, pt_weights in zip(model.layers[:-1], weight_list[:-1]):
        layer.set_weights(pt_weights)

# *** BEGIN SIAMESE NEURAL NETWORK CODE
# Modified from https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    # margin = math.sqrt(10)
    # margin = math.sqrt(100)
    # margin = math.sqrt(1000)
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def get_siamese(base_network, input_dim):
    # Create a siamese neural network with the provided base_network as two conjoined twins.
    # Load pretrained weights before calling this function.
    # First, remove the last layer (output layer) from base_network
    print("Siamese base Input shape: ", input_dim)
    base_model = Model(input = base_network.layers[0].input, output = base_network.layers[-2].output, name="BaseNetwork")
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name="Distance")([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    return model
# *** END SIAMESE NEURAL NETWORK CODE

def get_dense(hidden_layer_sizes, input_dim, activation_fcn='tanh'):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    x = inputs
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn)(x)
    return inputs, x

def get_ppitf(hidden_layer_sizes, input_dim, ppitf_groups_mat, activation_fcn='tanh'):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    # first hidden layer
    sparse_out = BioSparseLayer(activation=activation_fcn, input_output_mat=ppitf_groups_mat.transpose())(inputs)
    dense_out = Dense(100, activation=activation_fcn)(inputs)
    x = keras.layers.concatenate([sparse_out, dense_out])
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn)(x)
    return inputs, x

def get_nn_model(model_name, hidden_layer_sizes, input_dim, is_autoencoder, activation_fcn='tanh', ppitf_groups_mat=None, output_dim=None):
    print(hidden_layer_sizes)
    # First get the tensors from hidden layers
    if model_name == 'dense':
        in_tensors, hidden_tensors = get_dense(hidden_layer_sizes, input_dim, activation_fcn)
    elif model_name == 'ppitf':
        in_tensors, hidden_tensors = get_ppitf(hidden_layer_sizes, input_dim, ppitf_groups_mat, activation_fcn)
    else:
        raise ScrnaException("Bad neural network name: " + model_name)
    
    # Then add output layer on top
    if is_autoencoder:
        out_tensors = Dense(input_dim, activation=activation_fcn)(hidden_tensors)
    else:
        out_tensors = Dense(output_dim, activation='softmax')(hidden_tensors)
    model = Model(inputs=in_tensors, outputs=out_tensors)
    return model
