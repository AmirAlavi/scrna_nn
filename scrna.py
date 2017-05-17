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
    --epochs=<nepochs>      Number of epochs to train for [default: 20].
    --act=<activation_fcn>  Activation function to use for the layers [default: tanh].
    --data=<path>           Path to input data file [default: data/TPM_mouse_7_8_10_PPITF_gene_9437.txt].
    --sn                    Divide each sample by the total number of reads for that sample.
    --gs                    Subtract the mean and divide by standard deviation within each gene.
    --sgd_lr=<lr>           Learning rate for SGD [default: 0.1].
    --sgd_d=<decay>         Decay rate for SGD [default: 1e-6].
    --sgd_m=<momentum>      Momentum for SGD [default: 0.9].
    --sgd_nesterov          Use Nesterov momentum for SGD.
"""
from docopt import docopt

from util import ScrnaException
from neural_nets import get_nn_model
from sparse_optimizers import SparseSGD
from preprocessing import DataContainer

def train(args):
    # Get the training data
    data = DataContainer(args['--data'], args['--sn'], args['--gs'])
    X, y, label_strings_lookup = data.get_labeled_data()
    # Get the model architecture
    hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
    model = get_nn_model(args['<neural_net_architecture>'], hidden_layer_sizes, 100, args['--act'])
    # Set up the optimizer
    lr = float(args['--sgd_lr'])
    decay = float(args['--sgd_d'])
    momentum = float(args['--sgd_m'])
    sgd = SparseSGD(lr=lr, decay=decay, momentum=momentum, nesterov=args['--sgd_nesterov'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print("model compiled and ready for training")
    #print(model.summary())

if __name__ == '__main__':
    args = docopt(__doc__, version='scrna 0.1')
    print(args); print()
    try:
        if args['train']:
            train(args)
    except ScrnaException as e:
        msg = e.args[0]
        print("scrna excption: ", msg)
