# import pdb; pdb.set_trace()
import time
from os.path import join

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer


class DataContainer(object):
    """Parses and holds the input data table (gene expression file) in memory
    and provides access to various aspects of it.
    """
    def __init__(self, filepath, sample_normalize=False):
        self.filepath = filepath
        print('Reading in data from ', filepath)
        h5_store = pd.HDFStore(filepath)
        self.rpkm_df = h5_store['rpkm']
        self.labels_series = h5_store['labels'] if 'labels' in h5_store else None
        self.gene_symbols_series = h5_store['gene_symbols'] if 'gene_symbols' in h5_store else None
        self.accessions_series = h5_store['accessions'] if 'accessions' in h5_store else None
        h5_store.close()
        # Convert numeric to float32 for deep learning libraries
        self.rpkm_df = self.rpkm_df.apply(pd.to_numeric, errors='ignore', downcast='float')
        print("converted to float32")
        if sample_normalize:
            print("sample normalizing...")
            t0 = time.time()
            eps = np.finfo(np.float32).eps
            self.rpkm_df = self.rpkm_df.div(self.rpkm_df.sum(axis=1) + eps, axis=0)
            print("time to normalize: ", time.time() - t0)

    def get_gene_names(self):
        return self.gene_symbols_series.values

    def get_dataset_IDs(self):
        return self.accessions_series.values

    def get_labels(self):
        return self.labels_series.values

    def get_data(self):
        expression_mat = self.rpkm_df.values
        label_strings = self.labels_series.values
        uniq_label_strings, labels_as_int = np.unique(label_strings, return_inverse=True)
        return expression_mat, labels_as_int, uniq_label_strings

    def save_about_data(self, folder_to_save_in):
        """Save some descriptive info about this data to a text file.
        """
        with open(join(folder_to_save_in, 'about_data.txt'), 'w') as f:
            f.write("Source file: ", self.filepath, "\n")
            f.write("\nLabels present:\n")
            uniq, counts = np.unique(self.get_labels(), return_counts=True)
            f.write(uniq, "\n")
            f.write("\nCount for each label:\n")
            f.write(counts, "\n")


class DataContainer_old(object):
    # Deprecated in favor of cleaner, faster hdf5 version
    """Parses and holds the input data table (gene expression file) in memory
    and provides access to various aspects of it.
    """
    def __init__(self, filepath, sample_normalize=False, gene_standardize=False):
        self.filepath = filepath
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
            print("normalizing...")
            t0 = time.time()
            eps = np.finfo(np.float32).eps
            to_normalize = dataframe.iloc[:, 2:]
            dataframe.iloc[:, 2:] = to_normalize.div(to_normalize.sum(axis=1) + eps, axis=0)
            print("time to normalize: ", time.time() - t0)
        if gene_standardize:
            print("standardizing...")
            # Standardize each column by centering and having unit std
            t0 = time.time()
            for col in range(2, dataframe.shape[1]):
                std = dataframe.iloc[:, col].std(ddof=0)
                mean = dataframe.iloc[:, col].mean()
                dataframe.iloc[:, col] -= mean
                if std != 0:
                    dataframe.iloc[:, col] /= std
            print("time to standardize: ", time.time() - t0)

        self.dataframe = dataframe

    def get_gene_names(self):
        return self.dataframe.columns.values[2:]

    def get_all_dataset_IDs(self):
        return self.dataframe.loc[:, 'Dataset'].values.astype(int)

    def get_labeled_dataset_IDs(self):
        labeled_data = self.dataframe.loc[lambda df: df.Label != 'None', :]
        return labeled_data.loc[:, 'Dataset'].values.astype(int)

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

    def save_about_data(self, folder_to_save_in):
        """Save some descriptive info about this data to a text file.
        """
        with open(join(folder_to_save_in, 'about_data.txt'), 'w') as f:
            f.write("Source file: ", self.filepath, "\n")
            f.write("\nLabels present:\n")
            uniq, counts = np.unique(self.get_labeled_labels(), return_counts=True)
            f.write(uniq, "\n")
            f.write("\nCount for each label:\n")
            f.write(counts, "\n")
