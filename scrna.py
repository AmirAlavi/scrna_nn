"""Single-cell RNA-seq Analysis Pipeline.

Usage:
    scrna.py train (--nn=<nn_architecture> | --pca=<n_comp>) [<hidden_layer_sizes>...] [--out=<path> --data=<path>] [options]
    scrna.py reduce <trained_model_folder> [--out=<path> --data=<path> --save_meta]
    scrna.py visualize <reduced_data_file> [--out=<path> --ntypes=<ntypes> --nsamples=<nsamples> --title=<title>]
    scrna.py retrieval <query_data_file> <database_data_file> [--dist_metric=<metric> --out=<path>]
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

    GO_ppitf                  Combination of GO tree architecture and PPI/TF groupings in the 1st hidden layer.
                              Architecture specification is done through the arguments required for 'sparse' and
                              'GO' architectures.
                              - Can add Dense units in the first hidden layer to be concatenated with these
                                Sparse units with the '--with_dense' option.
                              - Additional hidden layers specified by '<hidden_layer_sizes>' are added as
                                Dense layers on top of the 1st hidden layer.

Options:
    -h --help                 Show this screen.
    --version                 Show version.
    --data=<path>             Path to input data file.
                              [default: data/lin_et_al_data/TPM_mouse_7_8_10_PPITF_gene_9437_T.txt]
    --out=<path>              Path to save output to. For training and retrieval this is a folder path.
                              For reduce this is a filepath (name of output file).
                              (trained models/reduced data/retrieval results).
                              'None' means that a time-stamped folder will
                              automatically be created. [default: None]

    "train" specific command options:
    --nn=<nn_architecture>    Train an instance of a nn_architecture neural network.
    --pca=<n_comp>            Fit a PCA model with n_comp principal components.
    --epochs=<nepochs>        Number of epochs to train for. [default: 100]
    --act=<activation_fcn>    Activation function to use for the layers.
                              [default: tanh]
    --sn                      Divide each sample by the total number of reads for
                              that sample.
    --valid=<valid_portion>   The portion of the training data to set aside for validation. Model is not
                              trained on this data. (only used if training a neural network model).
                              [default: 0.15]
    --gs                      Subtract the mean and divide by standard deviation
                              within each gene.
    --opt=<optimizer>         Optimizer to use. [default: 'sgd']
                              Available: 'sgd', 'rmsp'
                              TODO: Maybe make rmsp the default if it performs well?
    --sgd_lr=<lr>             Learning rate for SGD. [default: 0.1]
    --sgd_d=<decay>           Decay rate for SGD. [default: 1e-6]
    --sgd_m=<momentum>        Momentum for SGD. [default: 0.9]
    --sgd_nesterov            Use Nesterov momentum for SGD.
    --rmsp_lr=<lr>            Learning rate for RMSprop. [default: 0.001]
    --rmsp_rho=<rho>          Rho param for RMSprop. [default: 0.9]
    --rmsp_eps=<eps>          Fuzz factor for RMSprop. [default: 1e-08]
    --rmsp_decay=<decay>      Learning rate decay over each update for RMSprop. [default: 0.0]
    --sparse_groupings=<path> (For 'sparse' architecture) Path to file containing the genes
                              grouped to nodes for a sparse layer.
                              [default: data/mouse_ppitf_groups.txt]
    --fGO_ppitf_grps=<path,path> (For 'flatGO_ppitf' architecture) Paths to files containing the genes
                              grouped to nodes for sparse layers for a combined flatGO and ppitf architecture.
                              [default: data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt]
    --go_arch=<path>          Path to folder containing files that define a GO-based architecture.
                              [default: data/GO_lvls_arch_2_to_4]
    --with_dense=<num_units>  (For 'sparse' architecture) Number of Dense units to add in the same
                              layer as the Sparse layer. [default: 100]
    --pt=<weights_file>       Use initial weights from a pretrained model weights file.
    --ae                      Use an unsupervised autoencoder architecture.
    --siamese                 Uses a siamese neural network architecture, using
                              <nn_architecture> as the base network.
                              Using this flag has many implications, see code.
    --online_train=<n>        Dynamically generate hard pairs after n epochs for
                              siamese neural network training.
    --viz                     Visualize the data in the embedding space.

    "reduce" specific command options:
    --save_meta               Also save the metadata that was associated with the input data with the
                              reduced data (labels for the samples, accession numbers for the samples)

    "visualize" specific command options:
    --ntypes=<ntypes>         Number of different cell types to plot. Zero is interpreted as "all".
                              [default: 10]
    --nsamples=<nsamples>     Maximum number of samples of each selected cell type to plot.
                              [default: 100]
    --title=<title>           Title to use for plot. [default: "No title provided"]

    "retrieval" specific command options:
    --dist_metric=<metric>    Distance metric to use for nearest neighbors
                              retrieval [default: euclidean].
"""
# import pdb; pdb.set_trace()
import sys

from docopt import docopt

from train import train
from reduce import reduce
from visualize import visualize
from retrieval_test import retrieval_test
from util import ScrnaException


if __name__ == '__main__':
    print(sys.argv)
    args = docopt(__doc__, version='scrna 0.1')
    print(args); print()
    try:
        if args['train']:
            train(args)
        elif args['reduce']:
            reduce(args)
        elif args['visualize']:
            visualize(args)
        elif args['retrieval']:
            retrieval_test(args)
    except ScrnaException as e:
        msg = e.args[0]
        print("scrna exception: ", msg)
