#import pdb; pdb.set_trace()
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer


class DataContainer(object):
    """Parses and holds the input data table (gene expression file) in memory
    and provides access to various aspects of it.
    """
    def __init__(self, filepath, sample_normalize=False, gene_standardize=False):
        # Use pandas.read_csv to read in the file
        print('Reading in data from ', filepath)
        # TODO: Right now, because of format of files we currently have, the columns are mixed data types so we have to use low_memory=False
        dataframe = pd.read_csv(filepath, sep='\t', index_col=0, header=0, comment='#', low_memory=True)
        print("Read in data")
        # Currently, some input datasets have a Weight column that we want to
        # be Dataset ID number instead
        dataframe.rename(columns={'Weight': 'Dataset'}, inplace=True)
        # Column 0 is Label and Column 1 is Dataset. Columns after these are
        # the genes. Convert numeric columns to floats (downcast because
        # float64 might not be needed).
        dataframe = dataframe.apply(pd.to_numeric, errors='ignore', downcast='float')
        print("converted to float")
        if sample_normalize:
            # normalizer = Normalizer(norm='l1')
            # dataframe.iloc[:, 2:] = normalizer.fit_transform(dataframe.iloc[:, 2:])
            t0 = time.time()
            to_normalize = dataframe.iloc[:, 2:]
            dataframe.iloc[:, 2:] = to_normalize.div(to_normalize.sum(axis=1), axis=0)
            print("time to normalize: ", time.time() - t0)
            print("normalized")
        if gene_standardize:
            # Standardize each column by centering and having unit std
            t0 = time.time()
            # scaler = StandardScaler()
            # dataframe.iloc[:, 2:] = scaler.fit_transform(dataframe.iloc[:, 2:])
            to_std = dataframe.iloc[:, 2:]
            mean = to_std.mean()
            std = to_std.std(ddof=0)
            dataframe.iloc[:, 2:] = (to_std - mean).div(std, axis=0)
            print("time to standardize: ", time.time() - t0)
            print("standardized")
        self.dataframe = dataframe

    def get_gene_names(self):
        return self.dataframe.columns.values[2:]

    def get_all_dataset_IDs(self):
        return self.dataframe.loc[:, 'Dataset'].values

    def get_labeled_dataset_IDs(self):
        labeled_data = self.dataframe.loc[lambda df: df.Label != 'None', :]
        return labeled_data.loc[:, 'Dataset'].values

    def get_all_sample_IDs(self):
        #return self.dataframe.loc[:, 'Sample'].values
        pass

    def get_labeled_sample_IDs(self):
        #labeled_data = self.dataframe.loc[lambda df: df.Label != 'None', :]
        #return labeled_data.loc[:, 'Sample'].values
        pass

    def get_all_labels(self):
        return self.dataframe.loc[:, 'Label'].values

    def get_labeled_labels(self):
        labeled_data = self.dataframe.loc[lambda df: df.Label != 'None', :]
        return labeled_data.loc[:, 'Label'].values

    def get_labeled_data(self):
        print("getting labeled data")
        labeled_data = self.dataframe.loc[lambda df: df.Label != 'None', :]
        expression_mat = labeled_data.iloc[:, 2:].values
        label_strings = labeled_data.loc[:, 'Label'].values
        # Need to encode the labels as integers (like categorical data)
        # Can do this by getting a list of unique labels, then for each sample,
        # convert it's label to that label's index in this list. Oneliner:
        uniq_label_strings, labels_as_int = np.unique(label_strings, return_inverse=True)
        return expression_mat, labels_as_int, uniq_label_strings

    def get_unlabeled_data(self):
        pass

    def get_all_data(self):
        print("getting all data")
        expression_mat = self.dataframe.iloc[:, 2:].values
        # Some might be 'None'
        label_strings = self.dataframe.loc[:, 'Label'].values
        uniq_label_strings, labels_as_int = np.unique(label_strings, return_inverse=True)
        return expression_mat, labels_as_int, uniq_label_strings
