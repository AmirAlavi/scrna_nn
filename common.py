import numpy as np
from keras.utils import np_utils

from data_container import DataContainer

CLEAN_LABEL_SUBSET = ['2cell','4cell','ICM','zygote','8cell','ESC','lung','TE','thymus','spleen','HSC','neuron']

def modify_data_for_retrieval_test(data_container, test_labels):
    neuron_labels = ['cortex', 'CNS', 'brain']
    neuron_regexes = ['^.*'+label+'.*$' for label in neuron_labels]
    data_container.dataframe.replace(to_replace=neuron_regexes, value='neuron', inplace=True, regex=True)
    regexs = ['^.*'+label+'.*$' for label in test_labels]
    data_container.dataframe.replace(to_replace=regexs, value=test_labels, inplace=True, regex=True)

def preprocess_data(datacontainer):
    """Clean up the labels
    """
    modify_data_for_retrieval_test(datacontainer, CLEAN_LABEL_SUBSET)

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
