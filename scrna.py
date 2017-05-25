"""Single-cell RNA-seq Analysis Pipeline.

Usage:
    scrna.py train <neural_net_architecture> <hidden_layer_sizes>... [options]
    scrna.py evaluate
    scrna.py (-h | --help)
    scrna.py --version

Options:
    -h --help               Show this screen.
    --version               Show version.

    "train" command options:
    --epochs=<nepochs>      Number of epochs to train for. [default: 20]
    --act=<activation_fcn>  Activation function to use for the layers.
                            [default: tanh]
    --data=<path>           Path to input data file.
                            [default: data/TPM_mouse_7_8_10_PPITF_gene_9437.txt]
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
                            [default: ppi_tf_merge_cluster.txt]
"""
import time
from os.path import exists, join
from os import makedirs
import json
import pickle

from docopt import docopt
import numpy as np
from keras.utils import np_utils
#import theano
#theano.config.optimizer = 'None'

from util import ScrnaException
from neural_nets import get_nn_model, autoencoder_model_names, ppitf_model_names
from bio_knowledge import get_groupings_for_genes
from sparse_optimizers import SparseSGD
from data_container import DataContainer

def get_data(args):
    data = DataContainer(args['--data'], args['--sn'], args['--gs'])
    gene_names = data.get_gene_names()
    output_dim = None
    if args['<neural_net_architecture>'] in autoencoder_model_names:
        # Autencoder training is unsupervised, so we don't have to limit
        # ourselves to labeled samples
        X_clean, _, label_strings_lookup = data.get_all_data()
        # Add noise to the data:
        noise_level = 0.1
        X = X_clean + noise_level * np.random.normal(loc=0, scale=1, size=X.shape)
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
    if args['<neural_net_architecture>'] in ppitf_model_names:
        X = [X, X]
    return X, y, input_dim, output_dim, label_strings_lookup, gene_names

def get_model_architecture(args, input_dim, output_dim, gene_names):
    ppitf_groups_mat = None
    if args['<neural_net_architecture>'] in ppitf_model_names:
        _, _, ppitf_groups_mat = get_groupings_for_genes(args['--ppitf_groups'], gene_names)
        print("ppitf mat shape: ", ppitf_groups_mat.shape)
    hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
    return get_nn_model(args['<neural_net_architecture>'], hidden_layer_sizes, input_dim, args['--act'], ppitf_groups_mat, output_dim)

def get_optimizer(args):
    lr = float(args['--sgd_lr'])
    decay = float(args['--sgd_d'])
    momentum = float(args['--sgd_m'])
    return SparseSGD(lr=lr, decay=decay, momentum=momentum, nesterov=args['--sgd_nesterov'])

def train(args):
    # create a unique working directory for this model
    time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
    working_dir_path = "models/" + time_str + "_" + args['<neural_net_architecture>']
    if not exists(working_dir_path):
        makedirs(working_dir_path)
    print("loading data and setting up model...")
    X, y, input_dim, output_dim, label_strings_lookup, gene_names = get_data(args) # TODO: train/test/valid split
    print(X[0].shape)
    print(X[1].shape)
    model = get_model_architecture(args, input_dim, output_dim, gene_names)
    print(model.summary())
    sgd = get_optimizer(args)
    if args['<neural_net_architecture>'] in autoencoder_model_names:
        model.compile(loss='mean squared_error', optimizer=sgd)
    else:
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print("model compiled and ready for training")
    print("training model...")
    validation_data = (X, y) # For now, same as training data
    model.fit(X, y, epochs=int(args['--epochs']), verbose=2, validation_data=validation_data)
    print("saving model to folder: " + working_dir_path)
    model_path = join(working_dir_path, "model.p")
    print("model_path: ", model_path )
    #model.save(join(working_dir_path, "model.h5")) # TODO: Why doesn't this work?
    pickle.dump(model, open(model_path, 'wb'))
    with open(join(working_dir_path, "command_line_args.json"), 'w') as fp:
        json.dump(args, fp)

if __name__ == '__main__':
    args = docopt(__doc__, version='scrna 0.1')
    print(args); print()
    try:
        if args['train']:
            train(args)
    except ScrnaException as e:
        msg = e.args[0]
        print("scrna excption: ", msg)
