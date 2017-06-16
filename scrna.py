"""Single-cell RNA-seq Analysis Pipeline.

Usage:
    scrna.py train <neural_net_architecture> [<hidden_layer_sizes>...] [--sn --gs --data=<path> --out=<path> --act=<activation_fcn> --epochs=<nepochs> --sgd_lr=<lr> --sgd_d=<decay> --sgd_m=<momentum> --sgd_nesterov --ppitf_groups=<path> --pt --siamese]
    scrna.py reduce <trained_neural_net_folder> [--out=<path> --data=<path>]
    scrna.py retrieval <reduced_data_folder> [--dist_metric=<metric> --out=<path>]
    scrna.py (-h | --help)
    scrna.py --version

Options:
    -h --help               Show this screen.
    --version               Show version.
    --data=<path>           Path to input data file.
                            [default: data/TPM_mouse_7_8_10_PPITF_gene_9437_T.txt]
    --out=<path>            Path of folder to save output
                            (trained models/reduced data/retrieval results) to.
                            'None' means that a time-stamped folder will
                            automatically be created. [default: None]

    "train" specific command options:
    --epochs=<nepochs>      Number of epochs to train for. [default: 100]
    --act=<activation_fcn>  Activation function to use for the layers.
                            [default: tanh]
    --sn                    Divide each sample by the total number of reads for
                            that sample.
    --gs                    Subtract the mean and divide by standard deviation
                            within each gene.
    --sgd_lr=<lr>           Learning rate for SGD. [default: 0.1]
    --sgd_d=<decay>         Decay rate for SGD. [default: 1e-6]
    --sgd_m=<momentum>      Momentum for SGD. [default: 0.9]
    --sgd_nesterov          Use Nesterov momentum for SGD.
    --ppitf_groups=<path>   Path to file containing the TF groups and PPI
                            groups, each on separate lines.
                            [default: data/ppi_tf_merge_cluster.txt]
    --pt                    Use initial weights from a pretrained model.
    --siamese               Uses a siamese neural network architecture, using
                            <neural_net_architecture> as the base network.
                            Using this flag has many implications, see code.

    "retrieval" specific command options:
    --dist_metric=<metric>  Distance metric to use for nearest neighbors
                            retrieval [default: euclidean].

"""
# import pdb; pdb.set_trace()
import time
from os.path import exists, join
from os import makedirs
import json
import sys
from collections import defaultdict
from itertools import combinations
import random

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docopt import docopt
import numpy as np
import pandas as pd
from keras.utils import np_utils
import theano
from scipy.spatial import distance

from util import ScrnaException
import neural_nets as nn
from bio_knowledge import get_groupings_for_genes
from sparse_optimizers import SparseSGD
from data_container import DataContainer
from bio_sparse_layer import BioSparseLayer
import keras
keras.layers.BioSparseLayer = BioSparseLayer

def create_working_directory(out_path, parent, suffix):
    if out_path == 'None':
        time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
        out_path = join(parent ,time_str + "_" + suffix)
    if not exists(out_path):
        makedirs(out_path)
    return out_path

def create_data_pairs_diff_datasets(X, y, dataset_IDs, indices_lists, same_lim):
    pairs = []
    labels = []
    for label in range(len(indices_lists)):
        same_count = 0
        combs = combinations(indices_lists[label], 2)
        for comb in combs:
            # only add this pair to the list if each sample comes from a different dataset
            a_idx = comb[0]
            b_idx = comb[1]
            if dataset_IDs[a_idx] == dataset_IDs[b_idx]:
                continue
            pairs += [[ X[a_idx], X[b_idx] ]]
            labels += [1]
            same_count += 1
            # if same_count == same_lim:
            #     break
        # create the same number of different pairs
        diff_count = 0
        while diff_count < same_count:
            a_idx = random.choice(indices_lists[label])
            a = X[a_idx]
            diff_idx = random.randint(0, X.shape[0]-1)
            # Pick another sample that has a different label AND comes from a different dataset
            while(y[diff_idx] == label or dataset_IDs[a_idx] == dataset_IDs[diff_idx]):
                diff_idx = random.randint(0, X.shape[0]-1)
            b = X[diff_idx]
            pairs += [[ a, b ]]
            labels += [0]
            diff_count += 1
    print("Generated ", len(pairs), " pairs")
    print("Distribution of different and same pairs: ", np.bincount(labels))
    return np.array(pairs), np.array(labels)

def create_data_pairs(X, y, indices_lists, same_lim):
    pairs = []
    labels = []
    for label in range(len(indices_lists)):
        same_count = 0
        combs = combinations(indices_lists[label], 2)
        for comb in combs:
            pairs += [[ X[comb[0]], X[comb[1]] ]]
            labels += [1]
            same_count += 1
            if same_count == same_lim:
                break
        # create the same number of different pairs
        diff_count = 0
        while diff_count < same_count:
            a = X[random.choice(indices_lists[label])]
            diff_idx = random.randint(0, X.shape[0]-1)
            while(y[diff_idx] == label):
                diff_idx = random.randint(0, X.shape[0]-1)
            b = X[diff_idx]
            pairs += [[ a, b ]]
            labels += [0]
            diff_count += 1
    print("Generated ", len(pairs), " pairs")
    print("Distribution of different and same pairs: ", np.bincount(labels))
    return np.array(pairs), np.array(labels)

def build_indices_master_list(X, y):
    indices_lists = defaultdict(list) # dictionary of lists
    print(X.shape[0], "examples in dataset")
    for sample_idx in range(X.shape[0]):
        indices_lists[y[sample_idx]].append(sample_idx)
    return indices_lists

def get_data_for_siamese(data_container, args, same_lim):
    X, y, label_strings_lookup = data_container.get_labeled_data()
    print("bincount")
    print(np.bincount(y))
    indices_lists = build_indices_master_list(X, y)
    # X_siamese, y_siamese = create_data_pairs(X, y, indices_lists, same_lim)
    # print("X shape: ", X_siamese.shape)
    # print("y shape: ", y_siamese.shape)

    # Try with dataset-aware pair creation
    dataset_IDs = data_container.get_labeled_dataset_IDs()
    print("num samples: ", len(y))
    print("len(dataset_IDs): ", len(dataset_IDs))
    assert(len(dataset_IDs) == len(y))
    X_siamese, y_siamese = create_data_pairs_diff_datasets(X, y, dataset_IDs, indices_lists, same_lim)
    print("X shape: ", X_siamese.shape)
    print("y shape: ", y_siamese.shape)
    X_siamese = [ X_siamese[:, 0], X_siamese[:, 1] ]
    return X_siamese, y_siamese

def get_data(data_path, args):
    data = DataContainer(data_path, args['--sn'], args['--gs'])
    gene_names = data.get_gene_names()
    output_dim = None
    if args['<neural_net_architecture>'] in nn.autoencoder_model_names:
        # Autencoder training is unsupervised, so we don't have to limit
        # ourselves to labeled samples
        X_clean, _, label_strings_lookup = data.get_all_data()
        # Add noise to the data:
        noise_level = 0.1
        X = X_clean + noise_level * np.random.normal(loc=0, scale=1, size=X_clean.shape)
        X = np.clip(X, -1., 1.)
        # For autoencoders, the input is a noisy sample, and the networks goal
        # is to reconstruct the original sample, and so the output is the same
        # shape as the input, and our label vector "y" is no longer labels, but
        # is the uncorrupted samples
        y = X_clean
    else:
        # Supervised training:
        print("Supervised training")
        X, y, label_strings_lookup = data.get_labeled_data()
        output_dim = max(y) + 1
        y = np_utils.to_categorical(y, output_dim)
    input_dim = X.shape[1]
    # TODO: For ppitf models, since their architectures require a merge layer, the
    # input dimensions will look different, and get_data should take care of that
    if args['<neural_net_architecture>'] in nn.ppitf_model_names:
        X = [X, X]
        # if args['<neural_net_architecture>'] in nn.autoencoder_model_names:
        #     # The output shape of an autoencocer must match the input shape, so
        #     # we need to do the same as above for the y
        #     y = [y, y]
    return X, y, input_dim, output_dim, label_strings_lookup, gene_names, data

def get_model_architecture(args, input_dim, output_dim, gene_names):
    ppitf_groups_mat = None
    if args['<neural_net_architecture>'] in nn.ppitf_model_names:
        _, _, ppitf_groups_mat = get_groupings_for_genes(args['--ppitf_groups'], gene_names)
        print("ppitf mat shape: ", ppitf_groups_mat.shape)
    hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
    return nn.get_nn_model(args['<neural_net_architecture>'], hidden_layer_sizes, input_dim, args['--act'], ppitf_groups_mat, output_dim)

def get_optimizer(args):
    lr = float(args['--sgd_lr'])
    decay = float(args['--sgd_d'])
    momentum = float(args['--sgd_m'])
    return SparseSGD(lr=lr, decay=decay, momentum=momentum, nesterov=args['--sgd_nesterov'])

def compile_model(model, args, optimizer):
    loss = None
    metrics = None
    if args['<neural_net_architecture>'] in nn.autoencoder_model_names:
        loss = 'mean_squared_error'
    elif args['--siamese']:
        print("Using contrastive loss")
        loss = nn.contrastive_loss
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

def train(args):
    # create a unique working directory for this model
    working_dir_path = create_working_directory(args['--out'], "models/", args['<neural_net_architecture>'])
    print("loading data and setting up model...")
    # if args['--siamese']:
    #     get_data_for_siamese(args['--data'], args)
    X, y, input_dim, output_dim, label_strings_lookup, gene_names, data_container = get_data(args['--data'], args) # TODO: train/test/valid split
    print(X[0].shape)
    print(X[1].shape)
    model = get_model_architecture(args, input_dim, output_dim, gene_names)
    print(model.summary())
    if args['--pt']:
        hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
        nn.set_pretrained_weights(model, args['<neural_net_architecture>'], hidden_layer_sizes)
    if args['--siamese']:
        #base_net_input_dim = X.shape
        model = nn.get_siamese(model, input_dim)
        X, y = get_data_for_siamese(data_container, args, 500)
    sgd = get_optimizer(args)
    compile_model(model, args, sgd)
    print("model compiled and ready for training")
    print("training model...")
    validation_data = (X, y) # For now, same as training data
    history = model.fit(X, y, epochs=int(args['--epochs']), verbose=1, validation_data=validation_data)
    print("saving model to folder: " + working_dir_path)
    with open(join(working_dir_path, "command_line_args.json"), 'w') as fp:
        json.dump(args, fp)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(join(working_dir_path, 'loss.png'))
    architecture_path = join(working_dir_path, "model_architecture.json")
    weights_path = join(working_dir_path, "model_weights.p")
    nn.save_trained_nn(model, architecture_path, weights_path)
    #model.save(join(working_dir_path, "model.h5")) # TODO: Why doesn't this work?
    #pickle.dump(model, open(model_path, 'wb'))

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
    training_args_path = join(args['<trained_neural_net_folder>'], "command_line_args.json")
    with open(training_args_path, 'r') as fp:
        training_args = json.load(fp)
    # Must ensure that we use the same normalizations/sandardization from when model was trained
    X, y, input_dim, output_dim, label_strings_lookup, gene_names, data_container = get_data(args['--data'], training_args)
    print("output_dim ", output_dim)
    model_base_path = args['<trained_neural_net_folder>']
    architecture_path = join(model_base_path, "model_architecture.json")
    weights_path = join(model_base_path, "model_weights.p")
    model = nn.load_trained_nn(architecture_path, weights_path)
    #model = get_model_architecture(training_args, input_dim, output_dim, gene_names)
    #model = model_from_json
    #nn.load_model_weight_from_pickle(model, weights_path)
    #model.compile(optimizer='sgd', loss='mse') # arbitrary
    print(model.summary())
    # use the last hidden layer of the model as a lower-dimensional representation:
    last_hidden_layer = model.layers[-2]
    if training_args['<neural_net_architecture>'] in nn.ppitf_model_names:
        # these models have special input shape
        get_activations = theano.function([model.layers[0].layers[0].input, model.layers[0].layers[1].input], last_hidden_layer.output)
        X_transformed = get_activations(X[0], X[1])
    else:
        get_activations = theano.function([model.layers[0].input], last_hidden_layer.output)
        X_transformed = get_activations(X)
    print("reduced dimensions to: ", X_transformed.shape)
    working_dir_path = create_working_directory(args['--out'], "reduced_data/", training_args['<neural_net_architecture>'])
    save_reduced_data(working_dir_path, X_transformed, y, label_strings_lookup)
    save_reduced_data_to_csv(working_dir_path, X_transformed, data_container)
    with open(join(working_dir_path, "training_command_line_args.json"), 'w') as fp:
        json.dump(training_args, fp)

def modify_data_for_retrieval_test(data_container, test_labels):
    data_container.dataframe.replace(to_replace=['cortex', 'CNS', 'brain'], value='neuron', inplace=True)
    regexs = ['^.*'+label+'.*$' for label in test_labels]
    data_container.dataframe.replace(to_replace=regexs, value=test_labels, inplace=True, regex=True)

def average_precision(target, retrieved_list):
    total = 0
    correct = 0
    avg_precision = 0
    for r in retrieved_list:
        total += 1
        if r == target:
            correct += 1
            avg_precision += correct/float(total)
    if correct > 0:
        avg_precision /= float(correct)
    return avg_precision

def retrieval_test(args):
    training_args_path = join(args['<reduced_data_folder>'], "training_command_line_args.json")
    with open(training_args_path, 'r') as fp:
        training_args = json.load(fp)
    working_dir_path = create_working_directory(args['--out'], "retrieval_results/", training_args['<neural_net_architecture>'])
    # Load the reduced data
    data = DataContainer(join(args['<reduced_data_folder>'], "reduced.csv"))
    X, _, _ = data.get_labeled_data()
    testing_label_subset = ['2cell','4cell','ICM','zygote','8cell','ESC','lung','TE','thymus','spleen','HSC','neuron']
    modify_data_for_retrieval_test(data, testing_label_subset)

    datasetIDs = data.get_labeled_dataset_IDs()
    labels = data.get_labeled_labels()

    summary_csv_file = open(join(working_dir_path, "retrieval_summary.csv"), 'w')
    # Write out the file headers
    summary_csv_file.write('dataset,celltype,#cell,mean average precision\n')

    sorted_unique_datasetIDS = np.unique(datasetIDs)
    for dataset in sorted_unique_datasetIDS:
        # We will only compare samples from different datasets, so separate them
        current_ds_samples_indicies = np.where(datasetIDs == dataset)[0]
        current_ds_samples = X[current_ds_samples_indicies]
        other_ds_samples_indicies = np.where(datasetIDs != dataset)[0]
        other_ds_samples = X[other_ds_samples_indicies]
        distance_matrix = distance.cdist(current_ds_samples, other_ds_samples, metric=args['--dist_metric'])

        average_precisions_for_label = defaultdict(list)

        for index, distances in enumerate(distance_matrix):
            current_sample_idx = current_ds_samples_indicies[index]
            current_sample_label = labels[current_sample_idx]
            if current_sample_label not in testing_label_subset:
                continue
            sorted_distances_indicies = np.argsort(distances)

            # Count the total number of same label samples in other datasets
            total_same_label = 0
            for i in range(len(distances)):
                label = labels[other_ds_samples_indicies[i]]
                if label == current_sample_label:
                    total_same_label += 1

            retrieved_labels = []
            for retrieved_idx in sorted_distances_indicies[:total_same_label]:
                retrieved_labels.append(labels[other_ds_samples_indicies[retrieved_idx]])
            avg_precision = average_precision(current_sample_label, retrieved_labels)
            average_precisions_for_label[current_sample_label].append(avg_precision)

        for label, average_precisions in average_precisions_for_label.items():
            num_samples = len(average_precisions)
            summary_csv_file.write(str(dataset) + ',' + label + ',' + str(num_samples) + ',' + str(np.mean(average_precisions)) + '\n')

    summary_csv_file.close()

if __name__ == '__main__':
    print(sys.argv)
    args = docopt(__doc__, version='scrna 0.1')
    print(args); print()
    try:
        if args['train']:
            train(args)
        elif args['reduce']:
            reduce(args)
        elif args['retrieval']:
            retrieval_test(args)
    except ScrnaException as e:
        msg = e.args[0]
        print("scrna exception: ", msg)
