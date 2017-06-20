# import pdb; pdb.set_trace()
import pickle
import math

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Merge, Input, Lambda
#from keras.layers import Merge
from keras.utils import plot_model
from keras import backend as K

from bio_sparse_layer import BioSparseLayer
from util import ScrnaException

autoencoder_model_names = ['1layer_ae', '2layer_ppitf_ae', 'lin_ae_dense_100', 'lin_ae_dense_796']
ppitf_model_names = ['2layer_ppitf', '2layer_ppitf_ae', 'lin_ppitf_696.100', 'lin_ppitf_696.100_100', 'lin_pt_ppitf_696.100', 'lin_pt_ppitf_696.100_100', 'test']

def load_functional_model_weights(model, path):
    with open(path, 'rb') as fp:
        weight_list = pickle.load(fp)
    for layer, weights in zip(model.layers, weight_list):
        layer.set_weights(weights)

def save_functional_model_weights(model, path):
    weight_list = [layer.get_weights() for layer in model.layers]
    with open(path, 'wb') as fp:
        pickle.dump(weight_list, fp)

def save_model_weight_to_pickle(model, path):
    weight_list=[]
    for layer in model.layers:
        weights = layer.get_weights()
        l=len(weights)
        if l==0:
            # l==0 means that this was a special merge layer
            # (the concatenation layer in ppitf models) and
            # must be handled in a special way below
            l1= layer.layers[0].get_weights()
            l2= layer.layers[1].get_weights()
            weight_list.append([l,[l1,l2]])
        else:
            weight_list.append((l,weights))
    with open(path, 'wb') as fp:
        pickle.dump(weight_list, fp)

def load_model_weight_from_pickle(model, path):
    with open(path, 'rb') as fp:
        weight_list = pickle.load(fp)
    for layer, weights in zip(model.layers,weight_list):
        if weights[0] > 0:
            layer.set_weights(weights[1])
        else:
            # Special case for ppitf models
            layer.layers[0].layers[0].set_weights(weights[1][0])
            layer.layers[1].layers[0].set_weights(weights[1][1])
    return weight_list

def save_trained_nn(model, architecture_path, weights_path, is_functional):
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)
    if is_functional:
        save_functional_model_weights(model, weights_path)
    else:
        save_model_weight_to_pickle(model, weights_path)

def load_trained_nn(architecture_path, weights_path, is_functional):
    custom_obj = {'BioSparseLayer': BioSparseLayer}
    model = model_from_json(open(architecture_path).read(), custom_objects=custom_obj)
    print(model.summary())
    if is_functional:
        load_functional_model_weights(model, weights_path)
    else:
        load_model_weight_from_pickle(model, weights_path)
    # Must compile to use it for evaluatoin, but not going to train
    # this model, so we can compile it arbitrarily
    model.compile(optimizer='sgd', loss='mse')
    return model

def get_pretrained_weights(pt_file):
    weight_list = []
    with open(pt_file, 'rb') as weights_file:
        weight_list = pickle.load(weights_file)
    return weight_list

def set_pretrained_weights_ppitf_helper(model, pt_file, num_hidden_layers):
    weight_list = get_pretrained_weights(pt_file)
    # The first layer is special, because it is the concatenation of a 
    # BioSparseLayer and a Dense layer
    sparse_weights = weight_list[0][1][0]
    # model.layers[0].layers[0].set_weights(sparse_weights)
    model.layers[0].layers[0].layers[0].set_weights(sparse_weights)
    dense_weights = weight_list[0][1][1]
    model.layers[0].layers[1].set_weights(dense_weights)
    for i in range(1, num_hidden_layers):
        model.layers[i].set_weights(weight_list[i][1])

def lin_set_pretrain_weights(model):
    print("Setting pretrained weights using same method in Lin et al...")
    #pt_file100="pretrained_models/lin_ae_dense_100_weights.p"
    pt_file100="pretrained_models/exp_6.19_lin_ae_dense_100_weights.p"
    #pt_file796="pretrained_models/lin_ae_dense_796_weights.p"
    pt_file796="pretrained_models/exp_6.19_lin_ae_dense_796_weights.p"
    weight_list=[]
    for index,layer in enumerate(model.layers):
        weights = layer.get_weights()
        if index>0:
           continue
        l=len(weights)
        if l==0:
            print("setting for ppitf layer")
            pt_weight=get_pretrained_weights(pt_file100)[0][1]
            layer.layers[1].set_weights(pt_weight)
        if l>0:
            if weights[1].shape[0]==100:
                print("setting for dense100 layer")
                pt_weight=get_pretrained_weights(pt_file100)[0][1]
                layer.set_weights(pt_weight)
            else:
                print("setting for dense796 layer")
                pt_weight=get_pretrained_weights(pt_file796)[0][1]
                layer.set_weights(pt_weight)

def set_pretrained_weights_dense_helper(model, pt_file, num_hidden_layers):
    weight_list = get_pretrained_weights(pt_file)
    for i in range(num_hidden_layers):
        model.layers[i].set_weights(weight_list[i][1])

def set_pretrained_weights(model, model_name, hidden_layer_sizes):
    pt_2layer_ppitf = 'pretrained_models/pt_2layer_ppitf_ae_weights.p'
    pt_1layer_ppitf = 'pretrained_models/pt_1layer_ppitf_ae_weights.p'
    pt_1layer_dense_100 = 'pretrained_models/pt_1layer_dense_100_ae_weights.p'
    pt_1layer_dense_796 = 'pretrained_models/pt_1layer_dense_796_ae_weights.p'
    pt_2layer_dense_796_100 = 'pretrained_models/pt_2layer_dense_796_100_ae_weights.p'
    if model_name == '2layer_ppitf':
        set_pretrained_weights_ppitf_helper(model, pt_2layer_ppitf, 2)
    elif model_name == '1layer_ppitf':
        set_pretrained_weights_ppitf_helper(model, pt_1layer_ppitf, 1)
    elif model_name == 'dense':
        if len(hidden_layer_sizes) == 1:
            if hidden_layers_sizes[0] == 100:
                set_pretrained_weights_dense_helper(model, pt_1layer_dense_100, 1)
            elif hidden_layer_sizes[0] == 796:
                set_pretrained_weights_dense_helper(model, pt_1layer_dense_796, 1)
        elif len(hidden_layer_sizes) == 2:
            set_pretrained_weights_dense_helper(model, pt_2layer_dense_796_100, 2)
    else:
        raise ScrnaException("Pretrained model weights not available for this architecture!")
    print("Using pretrained weights")

def get_dense_autoencoder(hidden_layer_sizes, input_dim, activation_fcn='tanh'):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], input_dim=input_dim, activation=activation_fcn))
    for i in range(1, len(hidden_layer_sizes)):
        model.add(Dense(hidden_layer_sizes[i], activation=activation_fcn))
    model.add(Dense(input_dim, activation=activation_fcn))
    return model

def get_2layer_ppitf_autoencoder(second_hidden_layer_size, input_dim, ppitf_groups_mat, output_dim, activation_fcn='tanh'):
    left_branch = Sequential()
    left_branch.add(BioSparseLayer(input_dim=input_dim, activation=activation_fcn, input_output_mat=ppitf_groups_mat.transpose()))
    right_branch = Sequential()
    right_branch.add(Dense(100, input_dim=input_dim))
    merged = Merge([left_branch, right_branch], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Dense(second_hidden_layer_size, activation=activation_fcn))
    model.add(Dense(input_dim, activation=activation_fcn))
    return model

def get_1layer_ppitf(input_dim, ppitf_groups_mat, output_dim, activation_fcn='tanh'):
    left_branch = Sequential()
    left_branch.add(BioSparseLayer(input_dim=input_dim, activation=activation_fcn, input_output_mat=ppitf_groups_mat.transpose()))
    right_branch = Sequential()
    right_branch.add(Dense(100, input_dim=input_dim))
    merged = Merge([left_branch, right_branch], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Dense(output_dim, activation='softmax'))
    return model

def get_2layer_ppitf_new(second_hidden_layer_size, input_dim, ppitf_groups_mat, output_dim, activation_fcn='tanh'):
    inputs = Input(shape=(input_dim,))

    # hidden layers
    # first hidden layer
    sparse_out = BioSparseLayer(activation=activation_fcn, input_output_mat=ppitf_groups_mat.transpose())(inputs)
    dense_out = Dense(100)(inputs)
    merged = keras.layers.concatenate([sparse_out, dense_out])
    # second hidden layer
    dense_out = Dense(100, activation=activation_fcn, name="last_hidden_layer")(merged)

    # output layer
    classification = Dense(output_dim, activation='softmax')(dense_out)
    model = Model(inputs=inputs, outputs=classification)
    print("plotting ppitf model")
    plot_model(model, to_file='new_ppitf_architecture.png', show_shapes=True)
    return model

def get_2layer_ppitf(second_hidden_layer_size, input_dim, ppitf_groups_mat, output_dim, activation_fcn='tanh'):
    left_branch = Sequential()
    left_branch.add(BioSparseLayer(input_dim=input_dim, activation=activation_fcn, input_output_mat=ppitf_groups_mat.transpose()))
    right_branch = Sequential()
    right_branch.add(Dense(100, input_dim=input_dim))
    merged = Merge([left_branch, right_branch], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Dense(second_hidden_layer_size, activation=activation_fcn))
    model.add(Dense(output_dim, activation='softmax'))
    return model

def get_dense(hidden_layer_sizes, input_dim, output_dim, activation_fcn='tanh'):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], input_dim=input_dim, activation=activation_fcn))
    for i in range(1, len(hidden_layer_sizes)):
        model.add(Dense(hidden_layer_sizes[i], activation=activation_fcn))
    model.add(Dense(output_dim, activation='softmax'))
    return model

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
    print("siamese layers:")
    print(model.layers)
    print("base_model layers")
    print(model.layers[2].layers)
    print("plotting siamese model")
    plot_model(model, to_file='siamese_architecture.png', show_shapes=True)
    print("\nsummary of base net")
    m = model.layers[2]
    print(m.summary())
    return model
# *** END SIAMESE NEURAL NETWORK CODE

def get_nn_model(model_name, hidden_layer_sizes, input_dim, activation_fcn='tanh', ppitf_groups_mat=None, output_dim=None):
    print(hidden_layer_sizes)
    if model_name == '1layer_ae':
        return get_1layer_autoencoder(hidden_layer_sizes[0], input_dim, activation_fcn)
    elif model_name == '2layer_ppitf_ae':
        return get_2layer_ppitf_autoencoder(hidden_layer_sizes[0], input_dim, ppitf_groups_mat, output_dim, activation_fcn)
    elif model_name == '2layer_ppitf':
        return get_2layer_ppitf(100, input_dim, ppitf_groups_mat, output_dim, activation_fcn)
    elif model_name == 'dense':
        return get_dense(hidden_layer_sizes, input_dim, output_dim, activation_fcn)
    elif model_name == 'test':
        return get_2layer_ppitf_new(100, input_dim, ppitf_groups_mat, output_dim, activation_fcn)

    # Models from Lin et al paper
    elif model_name == 'lin_dense_100':
        return get_dense([100], input_dim, output_dim, activation_fcn)
    elif model_name == 'lin_dense_796':
        return get_dense([796], input_dim, output_dim, activation_fcn)
    elif model_name == 'lin_dense_796_100':
        return get_dense([796, 100], input_dim, output_dim, activation_fcn)
    elif model_name == 'lin_ppitf_696.100':
        return get_1layer_ppitf(input_dim, ppitf_groups_mat, output_dim, activation_fcn)
    elif model_name == 'lin_ppitf_696.100_100':
        return get_2layer_ppitf(100, input_dim, ppitf_groups_mat, output_dim, activation_fcn)
    elif model_name == 'lin_ae_dense_100':
        return get_dense_autoencoder([100], input_dim, activation_fcn)
    elif model_name == 'lin_ae_dense_796':
        return get_dense_autoencoder([796], input_dim, activation_fcn)
    elif model_name == 'lin_pt_dense_100':
        model = get_dense([100], input_dim, output_dim, activation_fcn)
        lin_set_pretrain_weights(model)
        return model
    elif model_name == 'lin_pt_dense_796':
        model = get_dense([796], input_dim, output_dim, activation_fcn)
        lin_set_pretrain_weights(model)
        return model
    elif model_name == 'lin_pt_dense_796_100':
        model = get_dense([796, 100], input_dim, output_dim, activation_fcn)
        lin_set_pretrain_weights(model)
        return model
    elif model_name == 'lin_pt_ppitf_696.100':
        model = get_1layer_ppitf(input_dim, ppitf_groups_mat, output_dim, activation_fcn)
        lin_set_pretrain_weights(model)
        return model
    elif model_name == 'lin_pt_ppitf_696.100_100':
        model = get_2layer_ppitf(100, input_dim, ppitf_groups_mat, output_dim, activation_fcn)
        lin_set_pretrain_weights(model)
        return model
    else:
        raise ScrnaException("Bad neural network name: " + model_name)
