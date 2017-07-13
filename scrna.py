"""Single-cell RNA-seq Analysis Pipeline.

Usage:
    scrna.py train <neural_net_architecture> [<hidden_layer_sizes>...] [--out=<path> --data=<path>] [options]
    scrna.py reduce <trained_neural_net_folder> [--out=<path> --data=<path>]
    scrna.py retrieval <reduced_data_folder> [--dist_metric=<metric> --out=<path>]
    scrna.py (-h | --help)
    scrna.py --version

Neural Net Architectures:
    dense                     Simple, fully connected layers, uses Keras's built-in Dense Layer.
                              '<hidden_layer_sizes>' specifies the number and sizes of the hidden layers

    sparse                    The connections between the input layer and the 1st hidden layer
                              are sparse (not all input nodes are connected to all 1st hidden layer
                              units, as in 'dense') using a custom Sparse layer. These connections are
                              specified through a 'grouping' file, see the '--sparse_groupings' option.
                              - Can add Dense units in the first hidden layer to be concatenated with these
                                Sparse units with the '--with_dense' option.
                              - Additional hidden layers specified by '<hidden_layer_sizes>' are added as
                                Dense layers on top of the 1st hidden layer.

    GO                        This architecture is based on the Gene Ontology. It is a tree, where each node
                              is a GO term, and its location in the tree and connections to ancestors is based
                              on the structure of GO (which is a DAG). This architecture is built up from multiple
                              Sparse layers, the connections defined by '--go_arch' option.'
                              - Can add Dense units in the first hidden layer to be concatenated with these
                                Sparse units with the '--with_dense' option.
                              - Additional hidden layers specified by '<hidden_layer_sizes>' are added as
                                Dense layers on top of the 1st hidden layer.

    flatGO_ppitf              Combination of Flattened GO groupings and PPI/TF groupings in the 1st hidden layer
                              of the neural net. Grouping files for both specified by '--fGO_ppitf_grps' option.
                              - Can add Dense units in the first hidden layer to be concatenated with these
                                Sparse units with the '--with_dense' option.
                              - Additional hidden layers specified by '<hidden_layer_sizes>' are added as
                                Dense layers on top of the 1st hidden layer.

    GO_ppitf                  (not implemented)

Options:
    -h --help                 Show this screen.
    --version                 Show version.
    --data=<path>             Path to input data file.
                              [default: data/TPM_mouse_7_8_10_PPITF_gene_9437_T.txt]
    --out=<path>              Path of folder to save output
                              (trained models/reduced data/retrieval results) to.
                              'None' means that a time-stamped folder will
                              automatically be created. [default: None]

    "train" specific command options:
    --epochs=<nepochs>        Number of epochs to train for. [default: 100]
    --act=<activation_fcn>    Activation function to use for the layers.
                              [default: tanh]
    --sn                      Divide each sample by the total number of reads for
                              that sample.
    --gs                      Subtract the mean and divide by standard deviation
                              within each gene.
    --sgd_lr=<lr>             Learning rate for SGD. [default: 0.1]
    --sgd_d=<decay>           Decay rate for SGD. [default: 1e-6]
    --sgd_m=<momentum>        Momentum for SGD. [default: 0.9]
    --sgd_nesterov            Use Nesterov momentum for SGD.
    --sparse_groupings=<path> (For 'sparse' architecture) Path to file containing the genes
                              grouped to nodes for a sparse layer.
                              [default: data/ppi_tf_merge_cluster.txt]
    --fGO_ppitf_grps=<path,path> (For 'flatGO_ppitf' architecture) Paths to files containing the genes
                              grouped to nodes for sparse layers for a combined flatGO and ppitf architecture.
                              [default: data/ppi_tf_merge_cluster.txt,data/flat_go300_groups.txt]
    --go_arch=<path>          Path to folder containing files that define a GO-based architecture
    --with_dense=<num_units>  (For 'sparse' architecture) Number of Dense units to add in the same
                              layer as the Sparse layer. [default: 100]
    --pt=<weights_file>       Use initial weights from a pretrained model weights file.
    --ae                      Use an unsupervised autoencoder architecture.
    --siamese                 Uses a siamese neural network architecture, using
                              <neural_net_architecture> as the base network.
                              Using this flag has many implications, see code.
    --online_train=<n>        Dynamically generate hard pairs after n epochs for
                              siamese neural network training.
    --viz                     Visualize the data in the embedding space.

    "retrieval" specific command options:
    --dist_metric=<metric>    Distance metric to use for nearest neighbors
                              retrieval [default: euclidean].

"""
# import pdb; pdb.set_trace()
import cProfile
import pickle
import time
from os.path import exists, join
from os import makedirs
import json
import sys
from collections import defaultdict, namedtuple
from itertools import combinations
import random

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docopt import docopt
import numpy as np
import pandas as pd
from keras.utils import np_utils, plot_model
import theano
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE

from util import ScrnaException
import neural_nets as nn
from bio_knowledge import get_adj_mat_from_groupings
from sparse_optimizers import SparseSGD
from data_container import DataContainer
from sparse_layer import Sparse
import keras
keras.layers.Sparse = Sparse

CLEAN_LABEL_SUBSET = ['2cell','4cell','ICM','zygote','8cell','ESC','lung','TE','thymus','spleen','HSC','neuron']
TESTING_LABEL_SUBSET = ['2cell','ESC','spleen','HSC','neuron']

def preprocess_data(datacontainer):
    """Clean up the labels
    """
    modify_data_for_retrieval_test(datacontainer, CLEAN_LABEL_SUBSET)

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
    print("num labels: ", len(indices_lists))
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
            if same_count == same_lim:
                break
        # create the same number of different pairs
        diff_count = 0
        while diff_count < (2 * same_count):
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
    print("Cleaning up the data first...")
    preprocess_data(data)
    gene_names = data.get_gene_names()
    output_dim = None
    if args['--ae']:
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
    return X, y, input_dim, output_dim, label_strings_lookup, gene_names, data

def get_model_architecture(args, input_dim, output_dim, gene_names):
    adj_mat = None
    go_other_levels_adj_mats = None
    flatGO_ppitf_adj_mats = None
    if args['<neural_net_architecture>'] == 'sparse':
        _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)
        print("Sparse layer adjacency mat shape: ", adj_mat.shape)
    elif args['<neural_net_architecture>'] == 'GO':
        # For now, we expect these file names
        # TODO: decouple file naming
        go_first_level_groupings_file = join(args['--go_arch'], 'GO_arch_first_level_groupings.txt')
        _, _, adj_mat = get_adj_mat_from_groupings(go_first_level_groupings_file, gene_names)
        print("(GO first level) Sparse layer adjacency mat shape: ", adj_mat.shape)
        go_other_levels_adj_mats_file = join(args['--go_arch'], 'GO_arch_other_levels_adj_mats.pickle')
        go_other_levels_adj_mats = pickle.load(go_other_levels_adj_mats_file)
    elif args['<neural_net_architecture>'] == 'flatGO_ppitf':
        _, _, flatGO_adj_mat = get_adj_mat_from_groupings(args['--fGO_ppitf_grps'].split(',')[0], gene_names)
        _, _, ppitf_adj_mat = get_adj_mat_from_groupings(args['--fGO_ppitf_grps'].split(',')[1], gene_names)
        flatGO_ppitf_adj_mats = [flatGO_adj_mat, ppitf_adj_mat]
    hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
    return nn.get_nn_model(args['<neural_net_architecture>'], hidden_layer_sizes, input_dim, args['--ae'], args['--act'], output_dim, adj_mat, go_other_levels_adj_mats, flatGO_ppitf_adj_mats, int(args['--with_dense']))

def get_optimizer(args):
    lr = float(args['--sgd_lr'])
    decay = float(args['--sgd_d'])
    momentum = float(args['--sgd_m'])
    return SparseSGD(lr=lr, decay=decay, momentum=momentum, nesterov=args['--sgd_nesterov'])

def compile_model(model, args, optimizer):
    loss = None
    metrics = None
    if args['--ae']:
        loss = 'mean_squared_error'
    elif args['--siamese']:
        print("Using contrastive loss")
        loss = nn.contrastive_loss
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

def plot_training_history(history, path):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path)

def get_hard_pairs(X, indices_lists, same_lim, ratio_hard_negatives, siamese_model=None):
    t0 = time.time()
    pairs = []
    labels = []
    if siamese_model:
        base_net = siamese_model.layers[2]
        get_embedding = theano.function([base_net.layers[0].input], base_net.layers[-1].output)
    # Initially generate pairs by going through each cell type, generate all same pairs, and
    # then list all the different samples sorted by their distance and choose the closest samples
    for cell_type in range(len(indices_lists)):
        # same pairs
        same_count = 0
        combs = list(combinations(indices_lists[cell_type], 2))
        np.random.shuffle(combs)
        #random.shuffle(combs)
        for comb in combs:
            pairs += [[ X[comb[0]], X[comb[1]] ]]
            labels += [1]
            same_count += 1
            if same_count == same_lim:
                break
        # hard different pairs
        # Pick a random representative of the current cell type
        rep_idx = random.choice(indices_lists[cell_type])
        rep  = X[rep_idx]
        if siamese_model:
            rep = get_embedding([rep])[0]
        # Get a list of all of the samples with different label
        all_different_indices = []
        for diff_cell_type in [x for x in range(len(indices_lists)) if x != cell_type]:
            all_different_indices += indices_lists[diff_cell_type]
        all_different_indices = np.array(all_different_indices)
        all_different = X[all_different_indices]
        if siamese_model:
            all_different = get_embedding(all_different)
        # Sort them by distance to the representative
        distances = euclidean_distances(all_different, [rep])
        #distances = np.linalg.norm(rep-all_different, axis=1) #slowest
        #distances = distance.cdist([rep], all_different, 'euclidean') #slow
        sorted_different_indices = all_different_indices[distances.argsort()]
        # Select pairs from these
        for i in range(same_count*ratio_hard_negatives):
            pairs += [[ X[rep_idx], X[sorted_different_indices[i]][0] ]]
            labels += [0]
    pairs = np.array(pairs)
    pairs = [ pairs[:, 0], pairs[:, 1] ]
    labels = np.array(labels)
    print("Picking new pairs took ", time.time()-t0, " seconds")
    return pairs, labels
    
def online_siamese_training(model, data_container, epochs, n, same_lim, ratio_hard_negatives):
    X_orig, y_orig, label_strings_lookup = data_container.get_labeled_data()
    indices_lists = build_indices_master_list(X_orig, y_orig)
    pairs = []
    labels = []
    # Initially generate pairs by going through each cell type, generate all same pairs, and
    # then list all the different samples sorted by their distance and choose the closest samples
    # print("profiling...")
    # cProfile.runctx('get_hard_pairs(X_orig, indices_lists, same_lim, ratio_hard_negatives)', globals={'get_hard_pairs':get_hard_pairs}, locals={'X_orig':X_orig, 'indices_lists':indices_lists, 'same_lim':same_lim, 'ratio_hard_negatives':ratio_hard_negatives}, filename='pairs_stats')
    # print("done profiling")
    X, y = get_hard_pairs(X_orig, indices_lists, same_lim, ratio_hard_negatives)
    print("Generated ", len(X[0]), " pairs")
    print("Distribution of different and same pairs: ", np.bincount(y))
    loss_list = []
    val_loss_list = []
    for epoch in range(0, epochs):
        print("Epoch ", epoch+1)
        epoch_hist = model.fit(X, y, epochs=1, verbose=1, validation_data=(X, y))
        loss_list.append(epoch_hist.history['loss'])
        val_loss_list.append(epoch_hist.history['val_loss'])
        if epoch % n == 0:
            # Get new pairs
            X, y = get_hard_pairs(X_orig, indices_lists, same_lim, ratio_hard_negatives, model)
    hist = {}
    hist['loss'] = loss_list
    hist['val_loss'] = val_loss_list
    History = namedtuple('History', ['history'])
    return History(history=hist)

def visualize_embedding(X, labels, path):
    print(X.shape)
    label_subset = {'HSC':'blue', '2cell':'green', 'spleen':'red', 'neuron':'cyan', 'ESC':'black'}
    # Only plot the subset of data
    subset_idx = []
    colors = []
    for i in range(len(labels)):
        if labels[i] in label_subset.keys():
            subset_idx.append(i)
            colors.append(label_subset[labels[i]])
    print("subset")
    print(len(subset_idx))
    subset_points = X[subset_idx]
    print(subset_points.shape)
    subset_labels = labels[subset_idx]
    tsne = TSNE(n_components=2, random_state=0)
    embedding = tsne.fit_transform(subset_points)
    plt.clf()
    plt.scatter(embedding[:,0], embedding[:,1], c=colors)
    plt.savefig(path)

def modify_data_for_retrieval_test(data_container, test_labels):
    neuron_labels = ['cortex', 'CNS', 'brain']
    neuron_regexes = ['^.*'+label+'.*$' for label in neuron_labels]
    data_container.dataframe.replace(to_replace=neuron_regexes, value='neuron', inplace=True, regex=True)
    regexs = ['^.*'+label+'.*$' for label in test_labels]
    data_container.dataframe.replace(to_replace=regexs, value=test_labels, inplace=True, regex=True)
    
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
    plot_model(model, to_file=join(working_dir_path, 'architecture.png'), show_shapes=True)
    print(model.summary())
    if args['--pt']:
        nn.set_pretrained_weights(model, args['--pt'])
    if args['--siamese']:
        model = nn.get_siamese(model, input_dim)
        plot_model(model, to_file='siamese_architecture.png', show_shapes=True)
    sgd = get_optimizer(args)
    compile_model(model, args, sgd)
    print("model compiled and ready for training")
    print("training model...")
    validation_data = (X, y) # For now, same as training data
    if args['--siamese'] and args['--online_train']:
        # Special online training (only an option for siamese nets)
        history = online_siamese_training(model, data_container, int(args['--epochs']), int(args['--online_train']), same_lim=2000, ratio_hard_negatives=2)
    else:
        # Normal training
        if args['--siamese']:
            X, y = get_data_for_siamese(data_container, args, 2000)
            validation_data = (X, y)
        history = model.fit(X, y, epochs=int(args['--epochs']), verbose=1, validation_data=validation_data)
    plot_training_history(history, join(working_dir_path, "loss.png"))
    print("saving model to folder: " + working_dir_path)
    with open(join(working_dir_path, "command_line_args.json"), 'w') as fp:
        json.dump(args, fp)
    architecture_path = join(working_dir_path, "model_architecture.json")
    weights_path = join(working_dir_path, "model_weights.p")
    if args['--siamese']:
        # For siamese nets, we only care about saving the subnetwork, not the whole siamese net
        model = model.layers[2] # For now, seems safe to assume index 2 corresponds to base net
    nn.save_trained_nn(model, architecture_path, weights_path)
    if args['--viz']:
        print("Visualizing...")
        X, _, _ = data_container.get_labeled_data()
        labels = data_container.get_labeled_labels()
        if args['--siamese']:
            last_hidden_layer = model.layers[-1]
        else:
            last_hidden_layer = model.layers[-2]
        get_activations = theano.function([model.layers[0].input], last_hidden_layer.output)
        X_embedded = get_activations(X)
        visualize_embedding(X_embedded, labels, join(working_dir_path, "tsne.png"))

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
    if training_args['--siamese']:
        print("Model was trained in a siamese architecture")
        last_hidden_layer = model.layers[-1]
    else:
        last_hidden_layer = model.layers[-2]
    get_activations = theano.function([model.layers[0].input], last_hidden_layer.output)
    X_transformed = get_activations(X)
    print("reduced dimensions to: ", X_transformed.shape)
    working_dir_path = create_working_directory(args['--out'], "reduced_data/", training_args['<neural_net_architecture>'])
    save_reduced_data(working_dir_path, X_transformed, y, label_strings_lookup)
    save_reduced_data_to_csv(working_dir_path, X_transformed, data_container)
    with open(join(working_dir_path, "training_command_line_args.json"), 'w') as fp:
        json.dump(training_args, fp)

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
    print("Cleaning up the data first...")
    preprocess_data(data)
    X, _, _ = data.get_labeled_data()

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
            if current_sample_label not in CLEAN_LABEL_SUBSET:
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
