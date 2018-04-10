# import pdb; pdb.set_trace()
import time
from os.path import join

import pandas as pd
import numpy as np


class DataContainer(object):
    """Parses and holds the input data table (gene expression file) in memory
    and provides access to various aspects of it.
    """
    def __init__(self, filepath, sample_normalize=False, feature_normalize=False, feature_mean=None, feature_std=None):
        self.filepath = filepath
        print('Reading in data from ', filepath)
        h5_store = pd.HDFStore(filepath)
        self.rpkm_df = h5_store['rpkm']
        self.labels_series = h5_store['labels'] if 'labels' in h5_store else None
        self.gene_symbols_series = h5_store['gene_symbols'] if 'gene_symbols' in h5_store else None
        self.accessions_series = h5_store['accessions'] if 'accessions' in h5_store else None
        self.true_ids_series = h5_store['true_ids'] if 'true_ids' in h5_store else None
        h5_store.close()
        self.rpkm_df.fillna(0, inplace=True) # Worries me that we have to do this...
        self.mean = feature_mean
        self.std = feature_std
        # Convert numeric to float32 for deep learning libraries
        self.rpkm_df = self.rpkm_df.apply(pd.to_numeric, errors='ignore', downcast='float')
        # Ensure the column names are stored as strings for campatability
        self.rpkm_df.columns = self.rpkm_df.columns.astype(str)
        print("converted to float32")
        eps = np.finfo(np.float32).eps
        if sample_normalize:
            print("sample normalizing...")
            t0 = time.time()

            self.rpkm_df = self.rpkm_df.div(self.rpkm_df.sum(axis=1) + eps, axis=0)
            print("time to normalize: ", time.time() - t0)
        elif feature_normalize:
            print("feature normalizing...")
            t0 = time.time()
            if self.mean is None and self.std is None:
                self.mean = self.rpkm_df.mean()
                self.std = self.rpkm_df.std(ddof=0)
            self.rpkm_df = (self.rpkm_df - self.mean) / (self.std + eps)
            print("time to normalize: ", time.time() - t0)

    def get_gene_names(self):
        return self.gene_symbols_series.values

    def get_dataset_IDs(self):
        return self.accessions_series.values

    def get_labels(self):
        return self.labels_series.values

    def get_true_ids(self):
        return self.true_ids_series.values

    def get_data(self):
        expression_mat = self.rpkm_df.values
        label_strings = self.labels_series.values
        uniq_label_strings, labels_as_int = np.unique(label_strings, return_inverse=True)
        return expression_mat, labels_as_int, uniq_label_strings

    def get_expression_mat(self):
        return self.rpkm_df.values

    def get_cell_ids(self):
        return self.rpkm_df.index.values

    def save_about_data(self, folder_to_save_in):
        """Save some descriptive info about this data to a text file.
        """
        with open(join(folder_to_save_in, 'about_data.txt'), 'w') as f:
            f.write("Source file: " + self.filepath + "\n")
            f.write("\nLabels present:\n")
            uniq, counts = np.unique(self.get_labels(), return_counts=True)
            f.write(str(uniq) + "\n")
            f.write("\nCount for each label:\n")
            f.write(str(counts) + "\n")
