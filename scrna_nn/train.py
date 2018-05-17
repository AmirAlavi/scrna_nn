# import pdb; pdb.set_trace()
import json
import pickle
import random
import time
import datetime
import math
from collections import defaultdict, namedtuple
from itertools import combinations
from os import makedirs
from os.path import join, exists

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import shuffle
from keras.utils import plot_model, np_utils, multi_gpu_model
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K

from .data_container import DataContainer
from . import util
from . import neural_nets as nn
from . import distances
from .bio_knowledge import get_adj_mat_from_groupings
from . import siamese
from . import triplet
from . import unsupervised_pt as pt
from . import callbacks
from . import losses_and_metrics

CACHE_ROOT = "_cache"
SIAM_CACHE = "siam_data"

def pretty_tdelta(tdelta):
    hours, rem = divmod(tdelta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return "{:2d} hours {:2d} mins {:2d} secs".format(hours, mins, secs)

def get_model_architecture(working_dir_path, args, input_dim, output_dim, gene_names):
    base_model = get_base_model_architecture(args, input_dim, output_dim, gene_names)
    #plot_model(base_model, to_file=join(working_dir_path, 'base_architecture.png'), show_shapes=True)
    print(base_model.summary())
    # Set pretrained weights, if any, before making into siamese
    if args['--init']:
        nn.set_pretrained_weights(base_model, args['--init'])
    if args['--freeze']:
        nn.freeze_layers(base_model, int(args['--freeze']))
    if args['--siamese']:
        model = nn.get_siamese(base_model, input_dim, args['--gn'])
        #plot_model(model, to_file=join(working_dir_path, 'siamese_architecture.png'), show_shapes=True)
    elif args['--triplet']:
        model = nn.get_triplet(base_model)
    else:
        model = base_model
    return model

def get_base_model_architecture(args, input_dim, output_dim, gene_names):
    """Possible options for neural network architectures are outlined in the '--help' command

    This function parses the user's options to determine what kind of architecture to construct.
    This could be a typical dense (MLP) architecture, a sparse architecture, or some combination.
    Users must provide an adjacency matrix for sparsely connected layers.
    """
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
        t0 = time.time()
        _, _, go_first_level_adj_mat = get_adj_mat_from_groupings(go_first_level_groupings_file, gene_names)
        print("get adj mat from groupings file took: ", time.time() - t0)
        print("(GO first level) Sparse layer adjacency mat shape: ", go_first_level_adj_mat.shape)
        go_other_levels_adj_mats_file = join(args['--go_arch'], 'GO_arch_other_levels_adj_mats.pickle')
        with open(go_other_levels_adj_mats_file, 'rb') as fp:
            go_other_levels_adj_mats = pickle.load(fp)
    elif args['--nn'] == 'flatGO_ppitf':
        _, _, flatGO_adj_mat = get_adj_mat_from_groupings(args['--fGO_ppitf_grps'].split(',')[0], gene_names)
        _, _, ppitf_adj_mat = get_adj_mat_from_groupings(args['--fGO_ppitf_grps'].split(',')[1], gene_names)
        flatGO_ppitf_adj_mats = [flatGO_adj_mat, ppitf_adj_mat]
    # elif args['--nn'] == 'GO_ppitf':
    #     _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)
        
    hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
    if args['--ae']:
        if args['--nn'] == 'GO':
            print("For GO autoencoder, doing 1st layer")
            adj_mat = go_first_level_adj_mat
        
    return nn.get_nn_model(args['--nn'], hidden_layer_sizes, input_dim, args['--ae'], args['--act'], output_dim, adj_mat, go_first_level_adj_mat, go_other_levels_adj_mats, flatGO_ppitf_adj_mats, int(args['--with_dense']), float(args['--dropout']))

def get_optimizer(args):
    if args['--opt'] == 'sgd':
        print("Using SGD optimizer")
        lr = float(args['--sgd_lr'])
        decay = float(args['--sgd_d'])
        momentum = float(args['--sgd_m'])
        return SGD(lr=lr, decay=decay, momentum=momentum, nesterov=args['--sgd_nesterov'])
    else:
        raise util.ScrnaException("Not a valid optimizer!")

def compile_model(model, args, optimizer):
    loss = None
    metrics = None
    if args['--ae']:
        loss = 'mean_squared_error'
    elif args['--siamese']:
        if args['--dynMarginLoss']:
            print("Using dynamic-margin contrastive loss")
            loss = losses_and_metrics.get_dynamic_contrastive_loss(float(args['--dynMargin']))
        else:
            print("Using contrastive loss")
            loss = nn.contrastive_loss
    elif args['--triplet']:
        batch_size = int(args['--batch_hard_P'])*int(args['--batch_hard_K'])
        loss = losses_and_metrics.get_triplet_batch_hard_loss(batch_size)
        frac_active_triplets = losses_and_metrics.get_frac_active_triplet_metric(batch_size)
        metrics = [frac_active_triplets]
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

def get_hard_pairs(X, indices_lists, same_lim, ratio_hard_negatives, siamese_model=None):
    t0 = time.time()
    pairs = []
    labels = []
    if siamese_model:
        base_net = siamese_model.layers[2]
        get_embedding = K.function([base_net.layers[0].input], [base_net.layers[-1].output])
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
            all_different = get_embedding([all_different])[0]
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
    indices_lists = util.build_indices_master_list(X_orig, y_orig)
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
        # TODO: allow user to set aside some for validation
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

def create_data_pairs(X, y, true_ids, indices_lists, same_lim, args):
    # cache_path = join(join(CACHE_ROOT, SIAM_CACHE), 'binary_200K')
    cache_path = join(CACHE_ROOT, SIAM_CACHE)
    cache_path = join(cache_path, args['--data']) # To make sure that we change cache when we change the dataset
    if exists(cache_path):
        print("Loading siamese data from cache...")
        pairs = np.load(join(cache_path, "siam_X.npy"))
        labels = np.load(join(cache_path, "siam_y.npy"))
        return pairs, labels
    pairs = []
    labels = []
    for label in range(len(indices_lists)):
        same_count = 0
        combs = combinations(indices_lists[label], 2)
        # TODO: should I shuffle the combs?
        for comb in combs:
            pairs += [[ X[comb[0]], X[comb[1]] ]]
            labels += [1]
            same_count += 1
            if same_count == same_lim:
                break
        # create the same number of different pairs
        diff_count = 0
        while diff_count < (2 * same_count):
            # pair of points (a, b) where a and b have diff labels
            a_idx = random.choice(indices_lists[label])
            a = X[a_idx]
            b_idx = random.randint(0, X.shape[0]-1)
            while y[b_idx] == label or true_ids[a_idx] == true_ids[b_idx]:
                b_idx = random.randint(0, X.shape[0]-1)
            b = X[b_idx]
            pairs += [[ a, b ]]
            labels += [0]
            diff_count += 1
    print("Generated ", len(pairs), " pairs")
    print("Distribution of different and same pairs: ", np.bincount(labels))
    pairs_np = np.array(pairs)
    labels_np = np.array(labels)
    makedirs(cache_path, exist_ok=True)
    np.save(join(cache_path, "siam_X"), pairs_np)
    np.save(join(cache_path, "siam_y"), labels_np)
    return pairs_np, labels_np

def get_distance(dist_mat, label_strings_lookup, max_dist, a_label, b_label, dist_fcn):
    a_str = label_strings_lookup[a_label]
    b_str = label_strings_lookup[b_label]
    dist = dist_mat[a_str][b_str]
    thresholded_dist = dist_fcn(dist, max_dist)
    return thresholded_dist

# def create_flexible_data_pairs(X, y, true_ids, indices_lists, same_lim, label_strings_lookup, args):
#     dist_mat_file = args['--distance_mat']
#     max_dist = int(args['--max_ont_dist'])
#     if args['--dist_fcn'] == 'linear':
#         print("Using linear distance decay")
#         dist_fcn = distances.linear_decay
#     elif args['--dist_fcn'] == 'exponential':
#         print("Using exponential distance decay")
#         dist_fcn = distances.exponential_decay
#     cache_path = join(join(CACHE_ROOT, SIAM_CACHE), args['--dist_fcn'])
#     cache_path = join(cache_path, args['--data']) # To make sure that we change cache when we change the dataset
#     if exists(cache_path):
#         print("Loading siamese data from cache...")
#         pairs = np.load(join(cache_path, "siam_X.npy"))
#         labels = np.load(join(cache_path, "siam_y.npy"))
#         return pairs, labels
#     print("Generating 'Flexible' pairs for siamese")
#     with open(dist_mat_file, 'rb') as f:
#         dist_mat_by_strings = pickle.load(f)
#     pairs = []
#     labels = []

#     for anchor_label, anchor_samples in indices_lists.items():
#         same_count = 0
#         combs = combinations(anchor_samples, 2)
#         # TODO: should I shuffle the combs?
#         for comb in combs:
#             pairs += [[ X[comb[0]], X[comb[1]] ]]
#             labels += [1]
#             same_count += 1
#             if same_count == same_lim:
#                 break
#         # create the different pairs
#         diff_count = 0
#         distance_lists = defaultdict(list)
#         for diff_label, diff_samples in indices_lists.items():
#             if diff_label == anchor_label:
#                 continue
#             dist = get_distance(dist_mat_by_strings, label_strings_lookup, max_dist, anchor_label, diff_label, dist_fcn)
#             for s in diff_samples:
#                 distance_lists[dist].append(s)
#         for distance, samples in distance_lists.items():
#             np.random.shuffle(samples)
#             num_pairs = min(len(samples), int((2*same_count)/max_dist))
#             for i in range(num_pairs):
#                 # select a random anchor sample
#                 anchor_idx = random.choice(anchor_samples)
#                 diff_idx = samples[i]
#                 while(true_ids[anchor_idx] == true_ids[diff_idx]):
#                     # for the current different sample, be sure they aren't the same underlying sample
#                     anchor_idx = random.choice(anchor_samples)
#                 anchor_vec = X[anchor_idx]
#                 diff_vec = X[diff_idx]
#                 pairs += [[ anchor_vec, diff_vec ]]
#                 labels += [distance]
#     print("Generated ", len(pairs), " pairs")
#     unique_labels, label_counts = np.unique(labels, return_counts=True)
#     print("Distribution of pairs labels: ")
#     print(unique_labels)
#     print(label_counts)
#     pairs_np = np.array(pairs)
#     labels_np = np.array(labels)
#     makedirs(cache_path)
#     np.save(join(cache_path, "siam_X"), pairs_np)
#     np.save(join(cache_path, "siam_y"), labels_np)
#     return pairs_np, labels_np

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

def get_data_for_siamese(data_container, args):
    X, y, label_strings_lookup = data_container.get_data()
    true_ids = data_container.get_true_ids()
    print("bincount")
    print(np.bincount(y))
    indices_lists = util.build_indices_master_list(X, y)
    # # Try with dataset-aware pair creation
    # dataset_IDs = data_container.get_dataset_IDs()
    # print("num samples: ", len(y))
    # print("len(dataset_IDs): ", len(dataset_IDs))
    # assert(len(dataset_IDs) == len(y))
    # X_siamese, y_siamese = create_data_pairs_diff_datasets(X, y, dataset_IDs, indices_lists, same_lim)
    same_lim = int(args['--same_lim'])
    if args['--dynMarginLoss']:
        X_siamese, y_siamese = siamese.create_flexible_data_pairs(X, y, true_ids, indices_lists, same_lim, label_strings_lookup, args)
    else:
        X_siamese, y_siamese = create_data_pairs(X, y, true_ids, indices_lists, same_lim, args)
    # Runs out of memory when trying to shuffle
    #X_siamese, y_siamese = shuffle(X_siamese, y_siamese) # Shuffle so that Keras's naive selection of validation data doesn't get all same class
    print("X shape: ", X_siamese.shape)
    print("y shape: ", y_siamese.shape)
    X_siamese = [ X_siamese[:, 0], X_siamese[:, 1] ]
    return X_siamese, y_siamese

def plot_accuracy_history(history, path):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(path)
    plt.close()
    
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

def get_data_for_testing(args, train_datacontainer, label_to_int_map):
    test_datacontainer = None
    if args['--gn']:
        test_datacontainer = DataContainer(args['--test_data'], sample_normalize=False, feature_normalize=True, feature_mean=train_datacontainer.mean, feature_std=train_datacontainer.std)
    else:
        test_datacontainer = DataContainer(args['--test_data'], sample_normalize=args['--sn'], feature_normalize=False)
    X = test_datacontainer.get_expression_mat()
    y_strings = test_datacontainer.get_labels()
    # Need to encode these labels with the same numbers as when we trained the model
    y = []
    for label_string in y_strings:
        y.append(np.expand_dims(label_to_int_map[label_string], 0))
    print(y[0].shape)
    y = np.concatenate(y, axis=0)
    print(y.shape)
    return X, y

def get_data_for_training(data_container, args):
    #print("Cleaning up the data first...")
    #preprocess_data(data)
    gene_names = data_container.get_gene_names()
    output_dim = None
    if args['--ae']:
        # Autoencoder training is unsupervised, so we don't have to limit
        # ourselves to labeled samples
        #X_clean = data_container.get_expression_mat()
        # Add noise to the data:
        #noise_level = 0.1
        #X = X_clean + noise_level * np.random.normal(loc=0, scale=1, size=X_clean.shape)
        #X = np.clip(X, -1., 1.)
        # For autoencoders, the input is a noisy sample, and the networks goal
        # is to reconstruct the original sample, and so the output is the same
        # shape as the input, and our label vector "y" is no longer labels, but
        # is the uncorrupted samples
        #y = X_clean
        X = data_container.get_expression_mat()
        y = X
        output_dim = X.shape[1]
        label_strings_lookup = None
        label_to_int_map = None
    else:
        # Supervised training:
        print("Supervised training")
        X, y, label_strings_lookup = data_container.get_data()
        output_dim = max(y) + 1
        if not args['--triplet']: # triplet net code needs labels that aren't 1-hot encoded
            print("One-hot enocoding")
            y = np_utils.to_categorical(y, output_dim)
        label_to_int_map = {}
        for i, label_string in enumerate(data_container.get_labels()):
            label_to_int_map[label_string] = y[i]
    input_dim = X.shape[1]
    print("Input dim: ", input_dim)
    print("Output dim: ", output_dim)
    return X, y, input_dim, output_dim, label_strings_lookup, gene_names, label_to_int_map

def train_pca_model(working_dir_path, args, data_container):
    print("Training a PCA model...")
    model = PCA(n_components=int(args['--pca']))
    X = data_container.get_expression_mat()
    model.fit(X)
    with open(join(working_dir_path, "pca.p"), 'wb') as f:
        pickle.dump(model, f)

def train_siamese_neural_net(model, args, data_container, callbacks_list):
    if args['--online_train']:
        # TODO: add callbacks option to online training
        history = online_siamese_training(model, data_container, int(args['--epochs']), int(args['--online_train']), same_lim=2000, ratio_hard_negatives=2)
    else:
        # same_lim =  750, 100K
        # same_lim = 1500, 200K
        # same_lim = 3000, 
        # same_lim = 3300, 400K
        # same_lim = 3500, 428K
        X, y = get_data_for_siamese(data_container, args) # this function shuffles the data too
        history = model.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)
    return history

def train_triplet_neural_net(model, args, X, y, callbacks_list):
    print(model.summary())
    train_frac = 1.0 - float(args['--valid'])
    split_idx = math.ceil(X.shape[0] * train_frac)
    embedding_dim = model.layers[-1].output_shape[1]
    P = int(args['--batch_hard_P'])
    K = int(args['--batch_hard_K'])
    num_batches = int(args['--num_batches'])
    train_data = triplet.TripletSequence(X[0:split_idx], y[0:split_idx], embedding_dim, P, K, num_batches)
    valid_data = triplet.TripletSequence(X[split_idx:], y[split_idx:], embedding_dim, P, K, num_batches)
    history = model.fit_generator(train_data, epochs=int(args['--epochs']), verbose=1, callbacks=callbacks_list, validation_data=valid_data)
    return history

def save_neural_net(working_dir_path, args, model):
    print("saving model to folder: " + working_dir_path)
    if args['--checkpoints']:
        path = join(working_dir_path, "last_model.h5")
    else:
        path = join(working_dir_path, "model.h5")
    if args['--siamese']:
        # For siamese nets, we only care about saving the subnetwork, not the whole siamese net
        model = model.layers[2] # For now, seems safe to assume index 2 corresponds to base net
    print("Model saved:\n\n\n")
    print(model.summary())
    nn.save_trained_nn(model, path)

def get_callbacks_list(working_dir_path, args):
    callbacks_list = []
    if args['--sgd_step_decay']:
        print("Using SGD Step Decay")
        lr_history = callbacks.StepLRHistory(float(args['--sgd_lr']), int(args['--sgd_step_decay']))
        lrate_sched = LearningRateScheduler(lr_history.get_step_decay_fcn())
        callbacks_list.extend([lr_history, lrate_sched])
    if int(args['--early_stop_pat']) >= 0:
        callbacks_list.append(EarlyStopping(monitor=args['--early_stop'], patience=int(args['--early_stop_pat']), verbose=1, mode='min'))
    if float(args['--early_stop_at_val']) >= 0:
        callbacks_list.append(callbacks.EarlyStoppingAtValue(monitor=args['--early_stop_at'], target=float(args['--early_stop_at_val']), verbose=1))
    if args['--checkpoints']:
        # checkpoints_folder = join(working_dir_path, 'checkpoints')
        # if not exists(checkpoints_folder):
        #     makedirs(checkpoints_folder)
        # callbacks_list.append(ModelCheckpoint(checkpoints_folder+"/model_{epoch:03d}-{val_loss:06.3f}.h5", monitor='val_loss', verbose=1, save_best_only=True))
        callbacks_list.append(ModelCheckpoint(working_dir_path+"/model.h5", monitor=args['--checkpoints'], verbose=1, save_best_only=True))
    if args['--loss_history']:
        callbacks_list.append(callbacks.LossHistory(working_dir_path))
    return callbacks_list
    
def train_neural_net(working_dir_path, args, data_container):
    print("Training a Neural Network model...")
    X, y, input_dim, output_dim, label_strings_lookup, gene_names, label_to_int_map = get_data_for_training(data_container, args)
    X, y = shuffle(X, y) # Shuffle so that Keras's naive selection of validation data doesn't get all same class
    opt = get_optimizer(args)

    
    ngpus = int(args['--ngpus'])
    if ngpus > 1:
        import tensorflow as tf
        with tf.device('/cpu:0'):
            template_model = get_model_architecture(working_dir_path, args, input_dim, output_dim, gene_names)
        model = multi_gpu_model(template_model, gpus=ngpus)
    else:
        template_model = get_model_architecture(working_dir_path, args, input_dim, output_dim, gene_names)
        model = template_model

    if args['--layerwise_pt']:
        hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
        if args['--nn'] == 'dense':
            if hidden_layer_sizes == [1136, 100]:
                pt.pretrain_dense_1136_100_model(model, input_dim, opt, X, working_dir_path, args)
            elif hidden_layer_sizes == [1136, 500, 100]:
                pt.pretrain_dense_1136_500_100_model(input_dim, opt, X, working_dir_path, args)
            else:
                raise util.ScrnaException("Layerwise pretraining not implemented for this architecture")
        elif args['--nn'] == 'sparse' and int(args['--with_dense']) == 100:
            if 'flat' in args['--sparse_groupings']:
                print('Using pretrained weights for FlatGO')
                if hidden_layer_sizes == [100]:
                    _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)
                    pt.pretrain_flatGO_400_100_model(input_dim, adj_mat, opt, X, working_dir_path, args)
                elif hidden_layer_sizes == [200, 100]:
                    _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)
                    pt.pretrain_flatGO_400_200_100_model(input_dim, adj_mat, opt, X, working_dir_path, args)
                else:
                    raise util.ScrnaException("Layerwise pretraining not implemented for this architecture")
            else:
                print("Using pretrained weights for PPITF")
                if hidden_layer_sizes == [100]:
                    _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)
                    pt.pretrain_ppitf_1136_100_model(input_dim, adj_mat, opt, X, working_dir_path, args)
                elif hidden_layer_sizes == [500, 100]:
                    _, _, adj_mat = get_adj_mat_from_groupings(args['--sparse_groupings'], gene_names)
                    pt.pretrain_ppitf_1136_500_100_model(input_dim, adj_mat, opt, X, working_dir_path, args)
                else:
                    raise util.ScrnaException("Layerwise pretraining not implemented for this architecture")

        elif args['--nn'] == 'GO' and int(args['--with_dense']) == 31:
            go_first_level_groupings_file = join(args['--go_arch'], 'GO_arch_first_level_groupings.txt')
            _, _, go_first_level_adj_mat = get_adj_mat_from_groupings(go_first_level_groupings_file, gene_names)
            go_other_levels_adj_mats_file = join(args['--go_arch'], 'GO_arch_other_levels_adj_mats.pickle')
            with open(go_other_levels_adj_mats_file, 'rb') as fp:
                go_other_levels_adj_mats = pickle.load(fp)
            pt.pretrain_GOlvls_model(input_dim, go_first_level_adj_mat, go_other_levels_adj_mats[0], go_other_levels_adj_mats[1], opt, X, working_dir_path, args)
        else:
            raise util.ScrnaException("Layerwise pretraining not implemented for this architecture")
        
    compile_model(model, args, opt)
    print("model compiled and ready for training")
    # Prep callbacks
    callbacks_list = get_callbacks_list(working_dir_path, args)
    print("training model...")
    t0 = datetime.datetime.now()
    if args['--siamese']:
        # Specially routines for training siamese models
        history = train_siamese_neural_net(model, args, data_container, callbacks_list)
    elif args['--triplet']:
        history = train_triplet_neural_net(model, args, X, y, callbacks_list)
        plt.figure()
        plt.semilogy(history.history['frac_active_triplet_metric'])
        plt.title('Fraction of active triplets per epoch')
        plt.ylabel('% active triplets')
        plt.xlabel('epoch')
        plt.savefig(join(working_dir_path, 'frac_active_triplets.png'))
        plt.close()
    else:
        history = model.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1-t0)
    print("Training neural net took " + time_str)
    with open(join(working_dir_path, "timing.txt"), 'w') as f:
        f.write(time_str + "\n")
    if not args['--ae'] and not args['--siamese'] and not args['--triplet']:
        plot_accuracy_history(history, join(working_dir_path, "accuracy.png"))
    if args['--test_data']:
        X_test, y_test = get_data_for_testing(args, data_container, label_to_int_map)
        print("Evaluating")
        eval_results = model.evaluate(x=X_test, y=y_test)
        with open(join(working_dir_path, "evaluation.txt"), 'w') as f:
            try:
                for metric, res in zip(model.metrics_names, eval_results):
                    print("{}\t{}".format(metric, res))
                    f.write("{}\t{}\n".format(metric, res))
            except TypeError:
                print(eval_results)
                f.write("{}\t{}\n".format(model.metrics_names, eval_results))
    save_neural_net(working_dir_path, args, template_model)
    # This code is an artifact, only works with an old dataset.
    # Needs some attention to make it work for the newer datasets.
    # if args['--viz']:
    #     print("Visualizing...")
    #     X, _, _ = data_container.get_data()
    #     labels = data_container.get_labels()
    #     if args['--siamese']:
    #         last_hidden_layer = model.layers[-1]
    #     else:
    #         last_hidden_layer = model.layers[-2]
    #     get_activations = theano.function([model.layers[0].input], last_hidden_layer.output)
    #     X_embedded = get_activations(X)
    #     visualize_embedding(X_embedded, labels, join(working_dir_path, "tsne.png"))
        
def train(args):
    model_type = args['--nn'] if args['--nn'] is not None else "pca"
    # create a unique working directory for this model
    working_dir_path = util.create_working_directory(args['--out'], "models/", model_type)
    with open(join(working_dir_path, "command_line_args.json"), 'w') as fp:
        json.dump(args, fp)
    print("loading data and setting up model...")
    data_container = DataContainer(args['--data'], sample_normalize=args['--sn'], feature_normalize=args['--gn'])
    if args['--gn']:
        # save the training data mean and std for later use with test data
        data_container.mean.to_pickle(join(working_dir_path, "mean.p"))
        data_container.std.to_pickle(join(working_dir_path, "std.p"))
    if args['--pca']:
        train_pca_model(working_dir_path, args, data_container)
    else:
        train_neural_net(working_dir_path, args, data_container)
