import pickle
from os.path import join
import json

import theano
import pandas as pd
import numpy as np

from util import create_working_directory
from common import get_data
import neural_nets as nn

def save_reduced_data_to_h5(out_folder, X_reduced, data_container):
    h5_store = pd.HDFStore(join(out_folder, "reduced.h5"))
    h5_store['rpkm'] = pd.DataFrame(data=X_reduced, index=data_container.rpkm_df.index)
    h5_store['labels'] = data_container.labels_series
    h5_store['accessions'] = data_container.accessions_series
    h5_store.close()
    #pass

def save_reduced_data_to_csv(out_folder, X_reduced, data_container):
    # Remove old data from the data container (but keep the Sample, Lable, and
    # Dataset columns)
    data = data_container.dataframe.loc[:, ['Label', 'Dataset']]
    reduced_data = pd.DataFrame(data=X_reduced, index=data.index)
    reduced_dataframe = pd.concat([data, reduced_data], axis=1)
    reduced_dataframe.to_csv(join(out_folder, "reduced.csv"), sep='\t', index_label="Sample")

def save_reduced_data(out_folder, X, y, label_strings_lookup):
    np.save(join(out_folder, "X"), X)
    np.save(join(out_folder, "y"), y)
    np.save(join(out_folder, "label_strings_lookup"), label_strings_lookup)

def reduce(args):
    training_args_path = join(args['<trained_model_folder>'], "command_line_args.json")
    with open(training_args_path, 'r') as fp:
        training_args = json.load(fp)
    # Must ensure that we use the same normalizations/sandardization from when model was trained
    X, y, input_dim, output_dim, label_strings_lookup, gene_names, data_container = get_data(args['--data'], training_args)
    print("output_dim ", output_dim)
    model_base_path = args['<trained_model_folder>']
    if training_args['--nn']:
        architecture_path = join(model_base_path, "model_architecture.json")
        weights_path = join(model_base_path, "model_weights.p")
        model = nn.load_trained_nn(architecture_path, weights_path)
        #model = get_model_architecture(training_args, input_dim, output_dim, gene_names)
        #model = model_from_json
        #nn.load_model_weight_from_pickle(model, weights_path)
        #model.compile(optimizer='sgd', loss='mse') # arbitrary
        print(model.summary())
        # use the last hidden layer of the model as a lower-dimensional representation:
        if training_args['--siamese']:
            print("Model was trained in a siamese architecture")
            last_hidden_layer = model.layers[-1]
        else:
            last_hidden_layer = model.layers[-2]
        get_activations = theano.function([model.layers[0].input], last_hidden_layer.output)
        X_transformed = get_activations(X)
    else:
        # Use PCA
        with open(join(model_base_path, "pca.p"), 'rb') as f:
            model = pickle.load(f)
        X_transformed = model.transform(X)
    print("reduced dimensions to: ", X_transformed.shape)
    model_type = training_args['--nn'] if training_args['--nn'] is not None else "pca"
    working_dir_path = create_working_directory(args['--out'], "reduced_data/", model_type)
    #save_reduced_data(working_dir_path, X_transformed, y, label_strings_lookup)
    #save_reduced_data_to_csv(working_dir_path, X_transformed, data_container)
    save_reduced_data_to_h5(working_dir_path, X_transformed, data_container)
    with open(join(working_dir_path, "training_command_line_args.json"), 'w') as fp:
        json.dump(training_args, fp)
