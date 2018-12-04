import argparse
import sys

from .. import analyze
from .. import reduce
from .. import retrieval_test
from .. import train
from .. import visualize


def save_cmd_args_to_file(path):
    with open(path, 'w') as f:
        f.write('\n'.join(sys.argv[1:]))


def load_cmd_args_from_file(path):
    parser = create_parser()
    return parser.parse_args(['@{}'.format(path)])


def create_parser():
    parser = argparse.ArgumentParser(
        description="Single-cell RNA-seq dimensionality reduction using " +
        "neural networks",
        fromfile_prefix_chars="@",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Add common options
    common_options_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    common_options_parser.add_argument(
        "--data",
        help="Path to input data. For 'train' command, this must be a folder " +
        "with train/valid/test files.",
        required=False)
    common_options_parser.add_argument(
        "--out",
        help="Path to save output to. For training and retrieval this is a " +
        "folder path.")

    # Add sub-commands
    subparsers = parser.add_subparsers(title="subcommands")
    # train
    parser_train = subparsers.add_parser(
        "train",
        help="Train a scRNA-seq dimensionality reduction model.",
        parents=[common_options_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train.set_defaults(func=train.train)
    parser_train.add_argument(
        "hidden_layer_sizes",
        help="List of hidden layer sizes (number of neurons per layer).",
        nargs='*')
    parser_train.add_argument(
        "--freeze",
        help="Freeze all but the last N layers (for fine tuning).",
        type=int)
    parser_train.add_argument(
        "--no_save",
        help="Do not save model weights.",
        action="store_true")

    parser_train.add_argument(
        "--no_eval",
        help="Do not run evaluation metrics after training.",
        action="store_true")        

    group_model_type = parser_train.add_mutually_exclusive_group()
    group_model_type.add_argument(
        "--pca",
        help="Fit a PCA model with specified number of principal components.",
        type=int)
    group_model_type.add_argument(
        "--nn",
        help="Train an instance of specified neural network architecture.",
        choices=[
            "dense",
            "sparse",
            "GO",
            "flatGO_ppitf",
            "GO_ppitf",
            "DAE",
            "sparseDAE"])

    group_normalization = parser_train.add_mutually_exclusive_group()
    group_normalization.add_argument(
        "--sn",
        help="Divide each sample by the total number of reads for that " +
        "sample.",
        action="store_true")
    group_normalization.add_argument(
        "--gn",
        help="Subtract the mean and divide by standard deviation within each " +
        "gene.",
        action="store_true")
    group_normalization.add_argument(
        "--mn",
        help="Scale values into a [min, max] range within each " +
        "gene.",
        action="store_true")

    parser_train.add_argument(
        "--minmax_min",
        help="Min of feature range for MinMax normalization.",
        type=float,
        default=-1)
    parser_train.add_argument(
        "--minmax_max",
        help="Max of feature range for MinMax normalization.",
        type=float,
        default=1)
    parser_train.add_argument(
        "--noise_level",
        help="Amount of corrupting noise to add to data " +
        "(used for DAE training).",
        type=float,
        default=0.1)
    parser_train.add_argument(
        "--valid",
        help="The portion of the training data to set aside for validation." +
        " Model is not trained on this data." +
        " (currently only unsed in usupervised pretraining models).",
        type=float,
        default=0.15)
    parser_train.add_argument(
        "--loss_history",
        help="Keep track of and plot loss history while training neural net.",
        action="store_true")

    group_arch = parser_train.add_argument_group('architecture')
    group_arch.add_argument(
        "--act",
        help="Activation function to use for the layers.",
        default="tanh")
    group_arch.add_argument(
        "--sparse_groupings",
        help="(For 'sparse' architecture) Path to file containing the genes " +
        "grouped to nodes for a sparse layer.")
    group_arch.add_argument(
        "--fGO_ppitf_grps",
        help="(For 'flatGO_ppitf' architecture) Paths to files containing the " +
        "genes grouped to nodes for sparse layers for a combined flatGO and" +
        " ppitf architecture.")
    group_arch.add_argument(
        "--go_arch",
        help="Path to folder containing files that define a GO-based " +
        "architecture.")
    group_arch.add_argument(
        "--with_dense",
        help="(For 'sparse' architecture) Number of Dense units to add in the" +
        " same layer as the Sparse layer.",
        type=int,
        default=0)
    group_arch.add_argument(
        "--init",
        help="Use initial weights from a pretrained weights file. If this " +
        "flag is not specified, random initialization is used.")
    group_arch.add_argument(
        "--layerwise_pt",
        help="Use greedy layer-wise pretraining to pretrain the model with " +
        "denoising autoencoders.",
        action="store_true")
    group_arch.add_argument(
        "--dropout",
        help="Use dropout layers to avoid overfitting. The location and " +
        "number of dropout layers depends on the architecture. " +
        "Rate of 0 denotes that no dropout layers should be added.",
        type=float,
        default=0)
    group_arch_reg = group_arch.add_mutually_exclusive_group()
    group_arch_reg.add_argument(
        "--l1_reg",
        help="Amount of L1 regularization to use (only on weights, not bias).",
        type=float,
        default=0)
    group_arch_reg.add_argument(
        "--l2_reg",
        help="Amount of L2 regularization to use (only on weights, not bias).",
        type=float,
        default=0)

    group_opt = parser_train.add_argument_group('optimizer')
    group_opt.add_argument(
        "--ngpus",
        help="Number of gpus to use. n > 1 is for data-parallel model" +
        " training (Only works with TensorFlow backend).",
        type=int,
        default=1)
    group_opt.add_argument(
        "--epochs",
        help="Number of epochs to train for.",
        type=int,
        default=100)
    group_opt.add_argument(
        "--batch_size",
        help="Number of samples per batch.",
        type=int,
        default=32)
    group_opt.add_argument(
        "--batches_per_epoch",
        help="Number of batches per training epoch.",
        type=int,
        default=None)
    group_opt.add_argument("--opt", help="Optimizer to use.",
                           choices=[
                               "sgd",
                               "adam",
                               "rmsprop"],
                           default="sgd")
    group_opt.add_argument(
        "--opt_lr",
        help="Learning rate for optimizer.",
        type=float,
        default=0.001)
    group_opt_decay = group_opt.add_mutually_exclusive_group()
    group_opt_decay.add_argument(
        "--sgd_d",
        help="Decay rate for SGD.",
        type=float,
        default=1e-6)
    group_opt_decay.add_argument(
        "--sgd_step_decay",
        help="Drop the learning rate by half after specified number of epochs.",
        type=int)
    group_opt.add_argument(
        "--sgd_m",
        help="Momentum for SGD.",
        type=float,
        default=0.9)
    group_opt.add_argument(
        "--sgd_nesterov",
        help="Use Nesterov momentum for SGD.",
        action="store_true")
    group_opt.add_argument(
        "--early_stop",
        help="Using early stopping in training by monitoring the specified " +
        "metric.",
        default="val_loss")
    group_opt.add_argument(
        "--early_stop_pat",
        help="Early stopping patience. Negative patience means no early " +
        "stopping.",
        type=int,
        default=-
        1)
    group_opt.add_argument(
        "--early_stop_at",
        help="Stop training when the specified metric reaches below or " +
        "equal to a target value.",
        default="val_loss")
    group_opt.add_argument(
        "--early_stop_at_val",
        help="Negative target value means no early stopping.",
        type=int,
        default=-1.0)
    group_opt.add_argument(
        "--checkpoints",
        help="Save best model (one with lowest score of specified metric)")

    group_siam = parser_train.add_argument_group('Siamese networks')
    group_siam.add_argument(
        "--siamese",
        help="Enable a siamese neural network architecture. Using this flag " +
        "has many implications, see code.",
        action="store_true")
    group_siam.add_argument(
        "--unif_diff",
        help="For Siamese pair selection, when selecting different pairs," +
        " select uniformly from n_buckets which stratify how different " +
        "the pairs are. 0 means select completely randomly, unconstrained.",
        type=int,
        default=0)
    group_siam.add_argument(
        "--same_lim",
        help="Maximum number of 'same' pairs to be generated for a cell-type.",
        type=int,
        default=750)
    group_siam.add_argument(
        "--diff_multiplier",
        help="For Siamese pair selection, when selecting different pairs, if " +
        "same_count pairs of 'same' points were generated, " +
        "generate (diff_multiplier * same_count) pairs of different points.",
        type=int,
        default=2)
    group_siam.add_argument(
        "--dynMarginLoss",
        help="Use a dynamic-margin Contrastive Loss for the Siamese training " +
        "which takes into account distances between cell-types " +
        "(rather than just binary, 0=different, 1=same)." +
        " Different types of distances are available, specified by the " +
        "required argument: ontology - distances are based on distances " +
        "between nodes in the Cell Ontology DAG (graph based). " +
        "These distances are the path lengths in the DAG converted " +
        "to an undirected graph. Need to be normalized using the" +
        " '--max_ont_dist' option. text-mined - distances are based on " +
        "co-occurance of terms in PubMed articles. These distances are " +
        "already normalized between 0 and 1.",
        choices=[
            "ontology",
            "text-mined"])
    group_siam.add_argument(
        "--dynMargin",
        help="Base margin value to use in contrastive loss.",
        type=int,
        default=1)
    group_siam.add_argument(
        "--dist_mat_file",
        help="A pickled, double-keyed dictionary of cell types whose values " +
        "are distances (this file provides the actual distances that " +
        "the user will use, must agree with the type of distance " +
        "specified in '--dynMarginLoss').")
    group_siam.add_argument(
        "--trnsfm_fcn",
        help="The type of transform function to use on top of raw similarity " +
        "values between cell types. A similarity of zero will always be " +
        "transformed to 0, and one will always be transformed to 1. Available transforms:" +
        " linear - linear decay from 1 to 0. " +
        "exponential - exponential growth from 0 to 1 " +
        "(lower bound to linear). " +
        "sigmoidal - sigmoidal (or tanh)-like function from 0 to 1 " +
        "(upper bound to linear). " +
        "binary - same pairs are 1, different pairs are all 0 " +
        "(will not use a dynamic-margin contrastive loss metric).",
        choices=[
            "linear",
            "exonential",
            "sigmoidal",
            "binary"],
        default="linear")
    group_siam.add_argument(
        "--trnsfm_fcn_param",
        help="A numerical constant necessary for some transform functions," +
        " allows you to tune their shape",
        type=float,
        default=1)
    group_siam.add_argument(
        "--max_ont_dist",
        help="The maximum distance allowed between nodes in the ontology " +
        "before their similarity is considered to be 0. " +
        "Only used for '--dynMarginLoss=ontology'.",
        type=int,
        default=4)
    group_siam.add_argument(
        "--online_train",
        help="Dynamically generate hard pairs after n epochs for " +
        "siamese neural network training.",
        type=int)

    group_trip = parser_train.add_argument_group('triplet networks')
    group_trip.add_argument(
        "--triplet",
        help="Uses 'batch-hard' triplet loss to train a triplet network.",
        action="store_true")
    group_trip.add_argument(
        "--plotter",
        help="Plot the triplet embeddings via t-SNE after each epoch " +
        "using the data in the folder specified.",
        default=None)
    group_trip.add_argument(
        "--pca_plotter",
        help="Plot the triplet embeddings via PCA instead of t-SNE." +
        " Pre-fitted PCA model in pickle file specified",
        default=None)
    group_trip.add_argument(
        "--plotter_int",
        help="Interval for plotting, plot every n epochs.",
        type=int,
        default=1)
    group_trip.add_argument(
        "--batch_hard_margin",
        help="Margin parameter to use in the loss calculation.",
        default='soft')
    group_trip.add_argument(
        "--batch_hard_P",
        help="'P' parameter in 'batch-hard' triplet loss " +
        "(number of classes to pick).",
        type=int,
        default=18)
    group_trip.add_argument(
        "--batch_hard_K",
        help="'K' parameter in 'batch-hard' triplet loss " +
        "(number of samples from each class to pick).",
        type=int,
        default=4)
    group_trip.add_argument(
        "--num_batches",
        help="Number of batches to be drawn in an epoch.",
        type=int,
        default=1000)
    group_trip.add_argument(
        "--num_batches_val",
        help="Number of validation batches to be drawn in an epoch.",
        type=int,
        default=100)

    # reduce
    parser_reduce = subparsers.add_parser(
        "reduce",
        help="Use a trained model to reduce dimensions (embed) scRNA-seq data.",
        parents=[common_options_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_reduce.set_defaults(func=reduce.reduce)
    parser_reduce.add_argument(
        "--save_meta",
        help="Also save the metadata that was associated with the input data " +
        "with the reduced data " +
        "(labels for the samples, accession numbers for the samples).",
        action="store_true")
    parser_reduce.add_argument(
        "trained_model_folder",
        help="Path to folder containing trained model.")

    # visualize
    parser_visualize = subparsers.add_parser(
        "visualize",
        help="Visualize reduced dimension data.",
        parents=[common_options_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_visualize.set_defaults(func=visualize.visualize)
    parser_visualize.add_argument(
        "reduced_data_file",
        help="Path to reduced dimenstion data (hdf5 dataframe).")
    parser_visualize.add_argument(
        "--ntypes",
        help="Number of different cell types to plot. " +
        "Zero is interpreted as 'all'.",
        type=int,
        default=10)
    parser_visualize.add_argument(
        "--nsamples",
        help="Maximum number of samples of each selected cell type to plot.",
        type=int,
        default=100)
    parser_visualize.add_argument(
        "--title",
        help="Title to use for plot.",
        default="No title provided")

    # retrieval
    parser_retrieval = subparsers.add_parser(
        "retrieval",
        help="Conduct a retrieval analysis experiment.",
        parents=[common_options_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_retrieval.set_defaults(func=retrieval_test.retrieval_test)
    parser_retrieval.add_argument(
        "query_data_file",
        help="Path to query samples (hdf5 dataframe).")
    parser_retrieval.add_argument(
        "database_data_file",
        help="Path to database samples (hdf5 dataframe).")
    parser_retrieval.add_argument(
        "--dist_metric",
        help="Distance metric to use for nearest neighbors retrieval.",
        default="euclidean")
    parser_retrieval.add_argument(
        "--similarity_type",
        help="Same as '--dynMarginLoss' from train command.",
        default="text-mined")
    parser_retrieval.add_argument(
        "--sim_mat_file",
        help="Same as '--dist_mat_file' from train command.")
    parser_retrieval.add_argument(
        "--sim_trnsfm_fcn",
        help="Same as '--trnsfm_fcn' from train command.",
        default="linear")
    parser_retrieval.add_argument(
        "--sim_trnsfm_param",
        help="Same as '--trnsfm_fcn_param' from train command.",
        type=float,
        default=1)
    parser_retrieval.add_argument(
        "--max_ont_path_len",
        help="See '--max_ont_dist' from the 'train' command. " +
        "This is the same thing.",
        type=int,
        default=4)
    parser_retrieval.add_argument(
        "--asymm_dist",
        help="Indicates that the similarity matrix is asymmetric.",
        action="store_true")

    # analyze
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze data (incomplete).",
        parents=[common_options_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_analyze.set_defaults(func=analyze.analyze)
    parser_analyze.add_argument(
        "trained_model_folder",
        help="Path to folder containing trained model.")
    parser_analyze.add_argument("query_data_file",
                                help="Path to query samples (hdf5 dataframe).")
    parser_analyze.add_argument(
        "database_data_file",
        help="Path to database samples (hdf5 dataframe).")

    return parser
