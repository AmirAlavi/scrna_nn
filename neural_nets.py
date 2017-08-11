# import pdb; pdb.set_trace()
import pickle
import math
import time

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Lambda
from keras import backend as K

from sparse_layer import Sparse
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
    custom_obj = {'Sparse': Sparse}
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

def get_sparse(hidden_layer_sizes, input_dim, adj_mat, activation_fcn='tanh', extra_dense_units=0):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    # first hidden layer
    sparse_out = Sparse(activation=activation_fcn, adjacency_mat=adj_mat)(inputs)
    if extra_dense_units > 0:
        dense_out = Dense(extra_dense_units, activation=activation_fcn)(inputs)
        x = keras.layers.concatenate([sparse_out, dense_out])
    else:
        x = sparse_out
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn)(x)
    return inputs, x

def get_GO(hidden_layer_sizes, input_dim, go_first_level_adj_mat, go_other_levels_adj_mats, activation_fcn='tanh', extra_dense_units=0):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    # first hidden layer
    # (Condsider entire GO tree (multi-level) as being in the 1st hidden layer)
    t0 = time.time()
    go_out = Sparse(activation=activation_fcn, adjacency_mat=go_first_level_adj_mat)(inputs)
    print("time to add GO lvl 1: ", time.time() - t0)
    for other_adj_mat in go_other_levels_adj_mats:
        t0 = time.time()
        go_out = Sparse(activation=activation_fcn, adjacency_mat=other_adj_mat)(go_out)
        print("time to add GO lvl: ", time.time() - t0)
    # Finished constructing GO tree
    if extra_dense_units > 0:
        dense_out = Dense(extra_dense_units, activation=activation_fcn)(inputs)
        x = keras.layers.concatenate([go_out, dense_out])
    else:
        x = go_out
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn)(x)
    return inputs, x

def get_flatGO_ppitf(hidden_layer_sizes, input_dim, flatGO_ppitf_adj_mats, activation_fcn='tanh', extra_dense_units=0):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    # first hidden layer
    sparse_flatGO_out = Sparse(activation=activation_fcn, adjacency_mat=flatGO_ppitf_adj_mats[0])(inputs)
    sparse_ppitf_out = Sparse(activation=activation_fcn, adjacency_mat=flatGO_ppitf_adj_mats[1])(inputs)
    if extra_dense_units > 0:
        dense_out = Dense(extra_dense_units, activation=activation_fcn)(inputs)
        x = keras.layers.concatenate([sparse_flatGO_out, sparse_ppitf_out, dense_out])
    else:
        x = keras.layers.concatenate([sparse_flatGO_out, sparse_ppitf_out])
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn)(x)
    return inputs, x

def get_GO_ppitf(hidden_layer_sizes, input_dim, ppitf_adj_mat, go_first_level_adj_mat, go_other_levels_adj_mats, activation_fcn='tanh', extra_dense_units=0):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    # first hidden layer
    ppitf_out = Sparse(activation=activation_fcn, adjacency_mat=ppitf_adj_mat)(inputs)
    # (Condsider entire GO tree (multi-level) as being in the 1st hidden layer)
    go_out = Sparse(activation=activation_fcn, adjacency_mat=go_first_level_adj_mat)(inputs)
    for other_adj_mat in go_other_levels_adj_mats:
        go_out = Sparse(activation=activation_fcn, adjacency_mat=other_adj_mat)(go_out)
    # Finished constructing GO tree
    if extra_dense_units > 0:
        dense_out = Dense(extra_dense_units, activation=activation_fcn)(inputs)
        x = keras.layers.concatenate([go_out, ppitf_out, dense_out])
    else:
        x = keras.layers.concatenate([go_out, ppitf_out])
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn)(x)
    return inputs, x

def get_nn_model(model_name, hidden_layer_sizes, input_dim, is_autoencoder, activation_fcn='tanh', output_dim=None, adj_mat=None, go_first_level_adj_mat=None, go_other_levels_adj_mats=None, flatGO_ppitf_adj_mats=None, extra_dense_units=0):
    print(hidden_layer_sizes)
    # First get the tensors from hidden layers
    if model_name == 'dense':
        in_tensors, hidden_tensors = get_dense(hidden_layer_sizes, input_dim, activation_fcn)
    elif model_name == 'sparse':
        in_tensors, hidden_tensors = get_sparse(hidden_layer_sizes, input_dim, adj_mat, activation_fcn, extra_dense_units)
    elif model_name == 'GO':
        in_tensors, hidden_tensors = get_GO(hidden_layer_sizes, input_dim, go_first_level_adj_mat, go_other_levels_adj_mats, activation_fcn, extra_dense_units)
    elif model_name == 'flatGO_ppitf':
        in_tensors, hidden_tensors = get_flatGO_ppitf(hidden_layer_sizes, input_dim, flatGO_ppitf_adj_mats, activation_fcn, extra_dense_units)
    elif model_name == 'GO_ppitf':
        in_tensors, hidden_tensors = get_GO_ppitf(hidden_layer_sizes, input_dim, adj_mat, go_first_level_adj_mat, go_other_levels_adj_mats, activation_fcn, extra_dense_units)
    else:
        raise ScrnaException("Bad neural network name: " + model_name)
    
    # Then add output layer on top
    if is_autoencoder:
        out_tensors = Dense(input_dim, activation=activation_fcn)(hidden_tensors)
    else:
        out_tensors = Dense(output_dim, activation='softmax')(hidden_tensors)
    model = Model(inputs=in_tensors, outputs=out_tensors)
    return model
