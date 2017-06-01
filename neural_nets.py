import pickle

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from keras.layers import Merge

from bio_sparse_layer import BioSparseLayer
from util import ScrnaException

autoencoder_model_names = ['1layer_ae']
ppitf_model_names = ['2layer_ppitf']

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

def save_trained_nn(model, architecture_path, weights_path):
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)
    save_model_weight_to_pickle(model, weights_path)

def load_trained_nn(architecture_path, weights_path):
    custom_obj = {'BioSparseLayer': BioSparseLayer}
    model = model_from_json(open(architecture_path).read(), custom_objects=custom_obj)
    print(model.summary())
    load_model_weight_from_pickle(model, weights_path)
    # Must compile to use it for evaluatoin, but not going to train
    # this model, so we can compile it arbitrarily
    model.compile(optimizer='sgd', loss='mse')
    return model

def get_1layer_autoencoder(hidden_layer_size, input_dim, activation_fcn='tanh'):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=input_dim, activation=activation_fcn))
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

def get_nn_model(model_name, hidden_layer_sizes, input_dim, activation_fcn='tanh', ppitf_groups_mat=None, output_dim=None):
    print(hidden_layer_sizes)
    if model_name == '1layer_ae':
        return get_1layer_autoencoder(hidden_layer_sizes[0], input_dim, activation_fcn)
    elif model_name == '2layer_ppitf_ae':
        return get_2layer_ppitf_autoencoder(hidden_layer_sizes[0], input_dim, ppitf_groups_mat, output_dim, activation_fcn)
    elif model_name == '2layer_ppitf':
        return get_2layer_ppitf(hidden_layer_sizes[0], input_dim, ppitf_groups_mat, output_dim, activation_fcn)
    else:
        raise ScrnaException("Bad neural network name: " + model_name)
