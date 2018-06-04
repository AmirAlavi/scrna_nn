# import pdb; pdb.set_trace()
import json
import pickle
from os import makedirs, remove
from os.path import join, dirname, exists

import pandas as pd
from keras import backend as K

from .data_manipulation.data_container import DataContainer
from .neural_network import neural_nets as nn


def save_reduced_data_to_h5(filename, X_reduced, data_container, save_metadata):
    if exists(filename):
        # delete file if it already exists because we want to overwrite it
        # (not easy to reclaim space in an existing hdf5 file)
        remove(filename)
    if dirname(filename) != '':
        makedirs(dirname(filename), exist_ok=True)
    h5_store = pd.HDFStore(filename)
    h5_store['rpkm'] = pd.DataFrame(data=X_reduced, index=data_container.rpkm_df.index)
    if save_metadata:
        print("saving metadata as well...")
        # Note: Does not make sense to save gene_symbols because our columns are no longer
        # genes, they are some reduced dimension.
        h5_store['labels'] = data_container.labels_series
        h5_store['accessions'] = data_container.accessions_series
    h5_store.close()


def _reduce_helper(trained_model_folder, data_to_reduce):
    training_args_path = join(trained_model_folder, "command_line_args.json")
    with open(training_args_path, 'r') as fp:
        training_args = json.load(fp)
    # Must ensure that we use the same normalizations/standardization from when model was trained
    mean = None
    std = None
    feature_normalize = False
    if training_args.gn:
        feature_normalize = True
        mean = pd.read_pickle(join(trained_model_folder, "mean.p"))
        std = pd.read_pickle(join(trained_model_folder, "std.p"))
    data_container = DataContainer(data_to_reduce, sample_normalize=training_args.sn, feature_normalize=feature_normalize, feature_mean=mean, feature_std=std)
    X = data_container.get_expression_mat()
    if training_args.nn:
        if training_args.triplet:
            triplet_batch_size = training_args.batch_hard_P*training_args.batch_hard_K
            model = nn.load_trained_nn(join(trained_model_folder, "model.h5"), triplet_loss_batch_size=triplet_batch_size)
        elif training_args.siamese:
            dynamic_margin=-1
            if '--dynMargin' in training_args:
                dynamic_margin = training_args.dynMargin
            model = nn.load_trained_nn(join(trained_model_folder, "model.h5"), dynamic_margin=dynamic_margin, siamese=True)
            if training_args.checkpoints:
                # HACK, TODO: account for this beforehand
                model = model.layers[2]
        else:
            model = nn.load_trained_nn(join(trained_model_folder, "model.h5"))
        print(model.summary())
        # use the last hidden layer of the model as a lower-dimensional representation:
        if training_args.siamese:
            print("Model was trained in a siamese architecture")
            last_hidden_layer = model.layers[-1]
        elif training_args.triplet:
            print("Model was trained in a triplet architecture")
            last_hidden_layer = model.layers[-1]
        else:
            last_hidden_layer = model.layers[-2]
        get_activations = K.function([model.layers[0].input], [last_hidden_layer.output])
        X_transformed = get_activations([X])[0]
    else:
        # Use PCA
        with open(join(trained_model_folder, "pca.p"), 'rb') as f:
            model = pickle.load(f)
        X_transformed = model.transform(X)
    print("reduced dimensions to: ", X_transformed.shape)
    return X_transformed, data_container
    
def reduce(args):
    X_transformed, original_data_container = _reduce_helper(args.trained_model_folder, args.data)
    save_reduced_data_to_h5(args.out, X_transformed, original_data_container, args.save_meta)
    # with open(join(working_dir_path, "training_command_line_args.json"), 'w') as fp:
    #     json.dump(training_args, fp)
