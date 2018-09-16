# import pdb; pdb.set_trace()
import time

import keras
from keras import backend as K
from keras import regularizers
from keras.layers import Dense, Input, Lambda, Dropout
from keras.models import Model, load_model

from . import losses_and_metrics
#from .sparse_layer import Sparse
from sparsely_connected_keras import Sparse
from tied_autoencoder_keras import DenseLayerAutoencoder
from ..util import ScrnaException


def save_trained_nn(model, model_path, weights_path):
    model.save(model_path)
    model.save_weights(weights_path)

def load_trained_nn(path, triplet_loss_batch_size=-1, dynamic_margin=-1, siamese=False):
    custom_objects={'Sparse': Sparse, 'DenseLayerAutoencoder': DenseLayerAutoencoder}
    if triplet_loss_batch_size >= 0:
        custom_objects['triplet_batch_hard_loss'] = losses_and_metrics.get_triplet_batch_hard_loss(triplet_loss_batch_size)
        custom_objects['frac_active_triplet_metric'] = losses_and_metrics.get_frac_active_triplet_metric(triplet_loss_batch_size)
    if siamese:
        if dynamic_margin == -1:
            dynamic_margin=1
        custom_objects['dynamic_contrastive_loss'] = losses_and_metrics.get_dynamic_contrastive_loss(dynamic_margin)
    print(custom_objects)
    return load_model(path, custom_objects=custom_objects)

# def get_pretrained_weights(pretrained_model_file):
#     pretrained_model = load_trained_nn(pretrained_model_file)
#     return [layer.get_weights() for layer in pretrained_model.layers]

def set_pretrained_weights(model, pretrained_weights_file):
    model.load_weights(pretrained_weights_file, by_name=True, skip_mismatch=True)
    print("Loaded pre-trained weights from: ", pretrained_weights_file)

def freeze_layers(model, n):
    """Freeze all but the last n layers
    """
    if len(model.layers) > n:
        for layer in model.layers[:-n]:
            print("Freezing layer: ", layer)
            layer.trainable = False

# *** BEGIN SIAMESE NEURAL NETWORK CODE
# Modified from https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def flexible_contrastive_loss(y_true, y_pred):
    """y_true is a float between 0 and 1.0, instead of binary.
    """
    #margin = 0.2
    margin = 1
    #margin = math.sqrt(10)
    #margin = math.sqrt(100)
    # margin = 6
    # margin = math.sqrt(1000)
    # margin = math.sqrt(1000)
    if y_true == 1:
        # Means that the distance in the ontology for this point was 0, exact same
        print("y_true == 1")
        return 0.5*K.square(y_pred)
    else:
        return 0.5*K.square(K.maximum((1-y_true)*margin - y_pred, 0))

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

# def triplet_loss(y_true, y_pred):
#     '''
#     '''

def get_siamese(base_network, input_dim, requires_norm=False):
    # Create a siamese neural network with the provided base_network as two conjoined twins.
    # Load pretrained weights before calling this function.
    # First, remove the last layer (output layer) from base_network
    print("Siamese base Input shape: ", input_dim)
    requires_norm = False
    if requires_norm:
        print("Adding L2 Norm layer to nework prior to euclidean distance (for Siamese)")
        features = base_network.layers[-2].output
        normed_features = Lambda(lambda x: K.l2_normalize(x, axis=1), output_shape=lambda input_shape: input_shape, name="l2_norm")(features)
        base_model = Model(inputs=base_network.layers[0].input, outputs=normed_features, name="BaseNetwork")
    else:
        base_model = Model(inputs=base_network.layers[0].input, outputs=base_network.layers[-2].output, name="BaseNetwork")

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name="Distance")([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    # if is_frozen and len(model.layers[2].layers) > 2:
    #     # Freeze the layers prior to the final embedding layer of the base network
    #     for layer in model.layers[2].layers[:-1]:
    #         print("Freezing layer: ", layer)
    #         layer.trainable = False
    return model
# *** END SIAMESE NEURAL NETWORK CODE

def get_triplet(base_network):
    embedding = base_network.layers[-2].output
    return Model(name="TripletNet", inputs=base_network.layers[0].input, outputs=embedding)

def get_dense(hidden_layer_sizes, input_dim, activation_fcn='tanh', dropout=0.0, regularization=None):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    x = inputs
    for size in hidden_layer_sizes:
        if dropout > 0:
            print("Using dropout layer")
            x = Dropout(dropout)(x)
        x = Dense(size, activation=activation_fcn, kernel_regularizer=regularization)(x)
    return inputs, x

def get_sparse(hidden_layer_sizes, input_dim, adj_df, activation_fcn='tanh', extra_dense_units=0, regularization=None):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    # first hidden layer
    adj_mat = adj_df.to_dense().values
    print("Sparse adj mat shape: {}".format(adj_mat.shape))
    sparse_out = Sparse(activation=activation_fcn, adjacency_mat=adj_mat, kernel_regularizer=regularization)(inputs)
    if extra_dense_units > 0:
        dense_out = Dense(extra_dense_units, activation=activation_fcn, kernel_regularizer=regularization)(inputs)
        x = keras.layers.concatenate([sparse_out, dense_out])
    else:
        x = sparse_out
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn, kernel_regularizer=regularization)(x)
    return inputs, x

def get_GO(hidden_layer_sizes, input_dim, GO_adj_dfs, activation_fcn='tanh', extra_dense_units=0, regularization=None):
    inputs = Input(shape=(input_dim,))
    go_out = inputs
    # Hidden layers
    # first hidden layer
    # (Consider entire GO tree (multi-level) as being in the 1st hidden layer)
    for adj_df in GO_adj_dfs:
        adj_mat = adj_df.to_dense().values
        go_out = Sparse(activation=activation_fcn, adjacency_mat=adj_mat, kernel_regularizer=regularization)(go_out)
    # Finished constructing GO tree
    if extra_dense_units > 0:
        dense_out = Dense(extra_dense_units, activation=activation_fcn, kernel_regularizer=regularization)(inputs)
        x = keras.layers.concatenate([go_out, dense_out])
    else:
        x = go_out
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn, kernel_regularizer=regularization)(x)
    return inputs, x

def get_flatGO_ppitf(hidden_layer_sizes, input_dim, flatGO_ppitf_adj_mats, activation_fcn='tanh', extra_dense_units=0, regularization=None):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    # first hidden layer
    sparse_flatGO_out = Sparse(activation=activation_fcn, adjacency_mat=flatGO_ppitf_adj_mats[0], kernel_regularizer=regularization)(inputs)
    sparse_ppitf_out = Sparse(activation=activation_fcn, adjacency_mat=flatGO_ppitf_adj_mats[1], kernel_regularizer=regularization)(inputs)
    if extra_dense_units > 0:
        dense_out = Dense(extra_dense_units, activation=activation_fcn, kernel_regularizer=regularization)(inputs)
        x = keras.layers.concatenate([sparse_flatGO_out, sparse_ppitf_out, dense_out])
    else:
        x = keras.layers.concatenate([sparse_flatGO_out, sparse_ppitf_out])
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn, kernel_regularizer=regularization)(x)
    return inputs, x

def get_GO_ppitf(hidden_layer_sizes, input_dim, ppitf_adj_mat, go_first_level_adj_mat, go_other_levels_adj_mats, activation_fcn='tanh', extra_dense_units=0, regularization=None):
    inputs = Input(shape=(input_dim,))
    # Hidden layers
    # first hidden layer
    ppitf_out = Sparse(activation=activation_fcn, adjacency_mat=ppitf_adj_mat, kernel_regularizer=regularization)(inputs)
    # (Condsider entire GO tree (multi-level) as being in the 1st hidden layer)
    go_out = Sparse(activation=activation_fcn, adjacency_mat=go_first_level_adj_mat, kernel_regularizer=regularization)(inputs)
    for other_adj_mat in go_other_levels_adj_mats:
        go_out = Sparse(activation=activation_fcn, adjacency_mat=other_adj_mat, kernel_regularizer=regularization)(go_out)
    # Finished constructing GO tree
    if extra_dense_units > 0:
        dense_out = Dense(extra_dense_units, activation=activation_fcn, kernel_regularizer=regularization)(inputs)
        x = keras.layers.concatenate([go_out, ppitf_out, dense_out])
    else:
        x = keras.layers.concatenate([go_out, ppitf_out])
    # other hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size, activation=activation_fcn, kernel_regularizer=regularization)(x)
    return inputs, x

def get_DAE(hidden_layer_sizes, input_dim, activation_fcn='tanh', dropout=0.0, regularization=None):
    inputs = Input(shape=(input_dim,))
    x = inputs
    x = DenseLayerAutoencoder(hidden_layer_sizes[0], activation=activation_fcn, kernel_regularizer=regularization)(x)
    return inputs, x

# def get_sparseDAE(input_dim, adj_mat, activation_fcn='tanh', extra_dense_units=0, regularization=None):
#     inputs = Input(shape=(input_dim,))
#     # Hidden layers
#     # first hidden layer
#     sparse_out = SparseLayerAutoencoder(activation=activation_fcn, adjacency_mat=adj_mat, kernel_regularizer=regularization)(inputs)
#     if extra_dense_units > 0:
#         dense_out = DenseLayerAutoencoder(extra_dense_units, activation=activation_fcn, kernel_regularizer=regularization)(inputs)
#         x = keras.layers.concatenate([sparse_out, dense_out])
#     else:
#         x = sparse_out
#     return inputs, x

def get_regularization(args):
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    reg = None
    if l1_reg > 0:
        reg = regularizers.l1(l1_reg)
    elif l2_reg > 0:
        reg = regularizers.l2(l2_reg)
    print('Using regularizer: {}'.format(reg))
    return reg

def get_nn_model(args, model_name, hidden_layer_sizes, input_dim, activation_fcn='tanh', output_dim=None, adj_mat=None, GO_adj_mats=None, extra_dense_units=0, dropout=0.0):
    # if is_autoencoder:
    #     # autoencoder architectures in a separate module for organizational purposes
    #     latent_size = None
    #     if hidden_layer_sizes is not None:
    #         latent_size = hidden_layer_sizes[0]
    #     return ae.get_ae_model(model_name, latent_size, input_dim, activation_fcn, adj_mat)
    is_autoencoder = False
    reg = get_regularization(args)
    print(hidden_layer_sizes)
    # First get the tensors from hidden layers
    if model_name == 'dense':
        in_tensors, hidden_tensors = get_dense(hidden_layer_sizes, input_dim, activation_fcn, dropout, reg)
    elif model_name == 'sparse':
        in_tensors, hidden_tensors = get_sparse(hidden_layer_sizes, input_dim, adj_mat, activation_fcn, extra_dense_units, reg)
    elif model_name == 'GO':
        in_tensors, hidden_tensors = get_GO(hidden_layer_sizes, input_dim, GO_adj_mats, activation_fcn, extra_dense_units, reg)
    elif model_name == 'DAE':
        in_tensors, hidden_tensors = get_DAE(hidden_layer_sizes, input_dim, activation_fcn, dropout, reg)
        is_autoencoder = True
    else:
        raise ScrnaException("Bad neural network name: " + model_name)
    # Then add output layer on top
    if is_autoencoder or output_dim is None: # (no explicit output layer required here)
        out_tensors = hidden_tensors
    else:
        out_tensors = Dense(output_dim, activation='softmax')(hidden_tensors)
    model = Model(inputs=in_tensors, outputs=out_tensors)
    return model
