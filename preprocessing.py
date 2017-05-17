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
        dataframe = pd.read_csv(filepath, sep='\t', index_col=0, header=0, comment='#')
        dataframe = dataframe.T
        # Column 0 is Label and Column 1 is Weight. Columns after these are
        # the genes. Select numeric columns (and convert them to floats)
        # (downcast because float64 might not be needed).
        converter = lambda series: pd.to_numeric(series, downcast='float')
        dataframe.iloc[:,1:] = dataframe.iloc[:,1:].apply(converter)
        if sample_normalize:
            normalizer = Normalizer(norm='l1')
            dataframe.iloc[:, 2:] = normalizer.fit_transform(dataframe.iloc[:, 2:])
        if gene_standardize:
            # Standardize each column by centering and having unit std
            scaler = StandardScaler()
            dataframe.iloc[:, 2:] = scaler.fit_transform(dataframe.iloc[:, 2:])
        self.dataframe = dataframe


    def get_labeled_data(self):
        print("getting labeled data")
        labeled_data = self.dataframe.loc[lambda df: df.Label != 'None', :]
        expression_mat = labeled_data.iloc[:, 2:].values
        print(expression_mat.shape)
        print(expression_mat)
        label_strings = labeled_data.loc[:, 'Label'].values
        # Need to encode the labels as integers (like categorical data)
        # Can do this by getting a list of unique labels, then for each sample,
        # convert it's label to that label's index in this list. Oneliner:
        uniq_label_strings, labels_as_int = np.unique(label_strings, return_inverse=True)
        return expression_mat, labels_as_int, uniq_label_strings

    def get_unlabeled_data(self):
        pass
