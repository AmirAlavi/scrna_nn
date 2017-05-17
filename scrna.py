"""Single-cell RNA-seq Analysis Pipeline.

Usage:
    scrna.py train <neural_net_architecture> <hidden_layer_sizes>... [--act=<activation_fcn> --epochs=<nepochs> --data=<path> --sn --gs]
    scrna.py evaluate
    scrna.py (-h | --help)
    scrna.py --version

Options:
    -h --help               Show this screen.
    --version               Show version.
    --epochs=<nepochs>      Number of epochs to train for [default: 20]. Only used
                            for 'train' command.
    --act=<activation_fcn>  Activation function to use for the layers [default: tanh].
    --data=<path>           Path to input data file [default: data/TPM_mouse_7_8_10_PPITF_gene_9437.txt].
    --sn                    Divide each sample by the total number of reads for that sample.
    --gs                    Subtract the mean and divide by standard deviation within each gene.

"""
from docopt import docopt

from util import ScrnaException
from neural_nets import get_nn_model
from preprocessing import DataContainer

def train(args):
    data = DataContainer(args['--data'], args['--sn'], args['--gs'])
    data.get_labeled_data()
    hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
    model = get_nn_model(args['<neural_net_architecture>'], hidden_layer_sizes, 100, args['--act'])
    print(model.summary())

if __name__ == '__main__':
    args = docopt(__doc__, version='scrna 0.1')
    print(args); print()
    try:
        if args['train']:
            train(args)
    except ScrnaException as e:
        msg = e.args[0]
        print("scrna excption: ", msg)
