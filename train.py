# import pdb; pdb.set_trace()
import cProfile
import pickle
import json
from os.path import join
from collections import defaultdict, namedtuple
import time
from itertools import combinations
import random

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from keras.utils import plot_model
import theano

from util import create_working_directory
from common import get_data
import neural_nets as nn
from bio_knowledge import get_adj_mat_from_groupings
from sparse_optimizers import SparseSGD

from sparse_layer import Sparse
import keras
keras.layers.Sparse = Sparse

def get_model_architecture(args, input_dim, output_dim, gene_names):
    adj_mat = None
    go_first_level_adj_mat = None
    go_other_levels_adj_mats = None
    flatGO_ppitf_adj_mats = None
    if args['--nn'] == 'sparse' or args['--nn'] == 'GO_ppitf':
        _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)
        print("Sparse layer adjacency mat shape: ", adj_mat.shape)
    if args['--nn'] == 'GO' or args['--nn'] == 'GO_ppitf':
        # For now, we expect these file names
        # TODO: decouple file naming
        go_first_level_groupings_file = join(args['--go_arch'], 'GO_arch_first_level_groupings.txt')
        _, _, go_first_level_adj_mat = get_adj_mat_from_groupings(go_first_level_groupings_file, gene_names)
        print("(GO first level) Sparse layer adjacency mat shape: ", go_first_level_adj_mat.shape)
        go_other_levels_adj_mats_file = join(args['--go_arch'], 'GO_arch_other_levels_adj_mats.pickle')
        go_other_levels_adj_mats = pickle.load(go_other_levels_adj_mats_file)
    elif args['--nn'] == 'flatGO_ppitf':
        _, _, flatGO_adj_mat = get_adj_mat_from_groupings(args['--fGO_ppitf_grps'].split(',')[0], gene_names)
        _, _, ppitf_adj_mat = get_adj_mat_from_groupings(args['--fGO_ppitf_grps'].split(',')[1], gene_names)
        flatGO_ppitf_adj_mats = [flatGO_adj_mat, ppitf_adj_mat]
    elif args['--nn'] == 'GO_ppitf':
        _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)
        
    hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
    return nn.get_nn_model(args['--nn'], hidden_layer_sizes, input_dim, args['--ae'], args['--act'], output_dim, adj_mat, go_first_level_adj_mat, go_other_levels_adj_mats, flatGO_ppitf_adj_mats, int(args['--with_dense']))

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

def build_indices_master_list(X, y):
    indices_lists = defaultdict(list) # dictionary of lists
    print(X.shape[0], "examples in dataset")
    for sample_idx in range(X.shape[0]):
        indices_lists[y[sample_idx]].append(sample_idx)
    return indices_lists

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
    X_orig, y_orig, label_strings_lookup = data_container.get_data()
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

def get_data_for_siamese(data_container, args, same_lim):
    X, y, label_strings_lookup = data_container.get_data()
    print("bincount")
    print(np.bincount(y))
    indices_lists = build_indices_master_list(X, y)
    # X_siamese, y_siamese = create_data_pairs(X, y, indices_lists, same_lim)
    # print("X shape: ", X_siamese.shape)
    # print("y shape: ", y_siamese.shape)

    # Try with dataset-aware pair creation
    dataset_IDs = data_container.get_dataset_IDs()
    print("num samples: ", len(y))
    print("len(dataset_IDs): ", len(dataset_IDs))
    assert(len(dataset_IDs) == len(y))
    X_siamese, y_siamese = create_data_pairs_diff_datasets(X, y, dataset_IDs, indices_lists, same_lim)
    print("X shape: ", X_siamese.shape)
    print("y shape: ", y_siamese.shape)
    X_siamese = [ X_siamese[:, 0], X_siamese[:, 1] ]
    return X_siamese, y_siamese

def plot_training_history(history, path):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path)

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

def train(args):
    model_type = args['--nn'] if args['--nn'] is not None else "pca"
    # create a unique working directory for this model
    working_dir_path = create_working_directory(args['--out'], "models/", model_type)
    print("loading data and setting up model...")
    # if args['--siamese']:
    #     get_data_for_siamese(args['--data'], args)
    X, y, input_dim, output_dim, label_strings_lookup, gene_names, data_container = get_data(args['--data'], args) # TODO: train/test/valid split
    print(X[0].shape)
    print(X[1].shape)
    if args['--pca']:
        print("Training a PCA model...")
        model = PCA(n_components=int(args['--pca']))
        model.fit(X)
        with open(join(working_dir_path, "pca.p"), 'wb') as f:
            pickle.dump(model, f)
    else:
        print("Training a Neural Network model...")
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
            X, _, _ = data_container.get_data()
            labels = data_container.get_labels()
            if args['--siamese']:
                last_hidden_layer = model.layers[-1]
            else:
                last_hidden_layer = model.layers[-2]
            get_activations = theano.function([model.layers[0].input], last_hidden_layer.output)
            X_embedded = get_activations(X)
            visualize_embedding(X_embedded, labels, join(working_dir_path, "tsne.png"))
