"""Single-cell RNA-seq Analysis Pipeline.

Usage:
    scrna.py train (--nn=<nn_architecture> | --pca=<n_comp>) <neural_net_architecture> [<hidden_layer_sizes>...] [--out=<path> --data=<path>] [options]
    scrna.py reduce <trained_model_folder> [--out=<path> --data=<path>]
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
    --out=<path>              Path of folder to save output
                              (trained models/reduced data/retrieval results) to.
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
    --gs                      Subtract the mean and divide by standard deviation
                              within each gene.
    --sgd_lr=<lr>             Learning rate for SGD. [default: 0.1]
    --sgd_d=<decay>           Decay rate for SGD. [default: 1e-6]
    --sgd_m=<momentum>        Momentum for SGD. [default: 0.9]
    --sgd_nesterov            Use Nesterov momentum for SGD.
    --sparse_groupings=<path> (For 'sparse' architecture) Path to file containing the genes
                              grouped to nodes for a sparse layer.
                              [default: data/mouse_ppitf_groups.txt]
    --fGO_ppitf_grps=<path,path> (For 'flatGO_ppitf' architecture) Paths to files containing the genes
                              grouped to nodes for sparse layers for a combined flatGO and ppitf architecture.
                              [default: data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt]
    --go_arch=<path>          Path to folder containing files that define a GO-based architecture.
                              [default: data/GO_4lvl_arch]]
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

    "retrieval" specific command options:
    --dist_metric=<metric>    Distance metric to use for nearest neighbors
                              retrieval [default: euclidean].

"""
# import pdb; pdb.set_trace()
from os.path import join
import json
import sys
from collections import defaultdict

from docopt import docopt
import numpy as np
from scipy.spatial import distance

from util import ScrnaException
from data_container import DataContainer
import common

TESTING_LABEL_SUBSET = ['2cell','ESC','spleen','HSC','neuron']

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
    model_type = training_args['--nn'] if training_args['--nn'] is not None else "pca"
    working_dir_path = create_working_directory(args['--out'], "retrieval_results/", model_type)
    # Load the reduced data
    data = DataContainer(join(args['<reduced_data_folder>'], "reduced.csv"))
    print("Cleaning up the data first...")
    common.preprocess_data(data)
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
            if current_sample_label not in common.CLEAN_LABEL_SUBSET:
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
