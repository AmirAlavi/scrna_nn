# import pdb; pdb.set_trace()
import time
from collections import defaultdict
from os import makedirs
from os.path import join, exists

import numpy as np
import pandas as pd
from keras.utils import np_utils, Sequence

from . import siamese


class DataContainer(object):
    """Parses and holds the input data table (gene expression file) in memory
    and provides access to various aspects of it.
    """
    def __init__(self, filepath, sample_normalize=False, feature_normalize=False, feature_mean=None, feature_std=None):
        self.sample_normalize = sample_normalize
        self.feature_normalize = feature_normalize
        self.mean = feature_mean
        self.std = feature_std
        self.splits = defaultdict(dict)
        self.add_split(filepath, 'train')
        self.label_to_int_map = None

    def _create_label_mapping(self):
        # Create a unique mapping of label string to integer, will be shared among all splits
        label_strings = self.splits['train']['labels_series'].values
        uniq_label_strings, y = np.unique(label_strings, return_inverse=True)
        self.label_to_int_map = {}
        for i, label_string in enumerate(label_strings):
            self.label_to_int_map[label_string] = y[i]
    
    def _normalize_split(self, split):
        eps = np.finfo(np.float32).eps
        if self.sample_normalize:
            print("sample normalizing...")
            t0 = time.time()
            self.splits[split]['rpkm_df'] = self.splits[split]['rpkm_df'].div(self.splits[split]['rpkm_df'].sum(axis=1) + eps, axis=0)
            print("time to normalize: ", time.time() - t0)
        elif self.feature_normalize:
            print("feature normalizing...")
            t0 = time.time()
            if split == 'train' and self.mean is None and self.std is None: # stats should already be in place for valid/test splits
                self.mean = self.splits[split]['rpkm_df'].mean()
                self.std = self.splits[split]['rpkm_df'].std(ddof=0)
            self.splits[split]['rpkm_df'] = (self.splits[split]['rpkm_df'] - self.mean) / (self.std + eps)
            print("time to normalize: ", time.time() - t0)
            
    def add_split(self, filepath, split):
        print('Reading in data from ', filepath)
        h5_store = pd.HDFStore(filepath)
        self.splits[split]['rpkm_df'] = h5_store['rpkm']
        self.splits[split]['labels_series'] = h5_store['labels'] if 'labels' in h5_store else None
        self.splits[split]['gene_symbols_series'] = h5_store['gene_symbols'] if 'gene_symbols' in h5_store else None
        if split != 'train':
            assert (np.array_equal(self.splits['train']['gene_symbols_series'].values, self.splits[split]['gene_symbols_series'].values)), 'New split does not have same columns as rest of data!'
        self.splits[split]['accessions_series'] = h5_store['accessions'] if 'accessions' in h5_store else None
        self.splits[split]['true_ids_series'] = h5_store['true_ids'] if 'true_ids' in h5_store else None
        h5_store.close()
        self.splits[split]['rpkm_df'].fillna(0, inplace=True) # Worries me that we have to do this...
        # Convert numeric to float32 for deep learning libraries
        self.splits[split]['rpkm_df'] = self.splits[split]['rpkm_df'].apply(pd.to_numeric, errors='ignore', downcast='float')
        print("converted to float32")
        # Ensure the column names are stored as strings for campatability
        self.splits[split]['rpkm_df'].columns = self.splits[split]['rpkm_df'].columns.astype(str)
        self._normalize_split(split)
    
    def get_gene_names(self):
        return self.splits['train']['gene_symbols_series'].values

    def get_dataset_IDs(self, split):
        return self.splits[split]['accessions_series'].values

    def get_labels(self, split='train'):
        return self.splits[split]['labels_series'].values

    def get_true_ids(self, split='train'):
        return self.splits[split]['true_ids_series'].values

    # def get_data(self):
    #     expression_mat = self.rpkm_df.values
    #     label_strings = self.labels_series.values
    #     uniq_label_strings, labels_as_int = np.unique(label_strings, return_inverse=True)
    #     return expression_mat, labels_as_int, uniq_label_strings

    def get_expression_mat(self, split='train'):
        return self.splits[split]['rpkm_df'].values

    def get_cell_ids(self, split='train'):
        return self.splits[split]['rpkm_df'].index.values

    def get_data_for_neural_net(self, split, one_hot=True):
        if self.label_to_int_map is None:
            self._create_label_mapping()
        X = self.splits[split]['rpkm_df'].values
        label_strings = self.splits[split]['labels_series'].values
        y = []
        for label in label_strings:
            y.append(self.label_to_int_map[label])
        output_dim = len(self.label_to_int_map)
        if one_hot:
            y = np_utils.to_categorical(y, output_dim)
        return X, y

    def get_data_for_neural_net_unsupervised(self, split, noise_level):
        X_clean = self.splits[split]['rpkm_df'].values
        X = X_clean + noise_level * np.random.normal(loc=0, scale=1, size=X_clean.shape)
        return X, X_clean

    def get_in_out_dims(self):
        if self.label_to_int_map is None:
            self._create_label_mapping()
        return self.splits['train']['rpkm_df'].shape[1], len(self.label_to_int_map)
    # def save_about_data(self, folder_to_save_in):
    #     """Save some descriptive info about this data to a text file.
    #     """
    #     with open(join(folder_to_save_in, 'about_data.txt'), 'w') as f:
    #         f.write("Source file: " + self.filepath + "\n")
    #         f.write("\nLabels present:\n")
    #         uniq, counts = np.unique(self.get_labels(), return_counts=True)
    #         f.write(str(uniq) + "\n")
    #         f.write("\nCount for each label:\n")
    #         f.write(str(counts) + "\n")

    def _create_siamese_data_split(self, args, split):
        CACHE_ROOT = '_cache'
        SIAM_CACHE = 'siam_data'
        # First check if we have already cached the siamese data for this configuration
        # Build a configuration string (a string that uniquely identifies a configuration)
        normalization = ""
        if args.sn:
            normalization = "sn"
        elif args.gn:
            normalization = "gn"
        config_string = '_'.join([args.data, normalization, args.dynMarginLoss, args.dist_mat_file, args.trnsfm_fcn, str(args.trnsfm_fcn_param), str(args.unif_diff), str(args.same_lim), str(args.diff_multiplier)])
        cache_path = join(join(join(CACHE_ROOT, SIAM_CACHE), config_string), split)
        if exists(cache_path):
            print("Loading siamese data from cache...")
            siam_X = np.load(join(cache_path, "siam_X.npy"))
            print('Siamese X shape:')
            print(siam_X.shape)
            self.splits[split]['siam_X'] = [ siam_X[:, 0], siam_X[:, 1] ]
            self.splits[split]['siam_y'] = np.load(join(cache_path, "siam_y.npy"))
        else:
            # If cached data doesn't exist, we have to make it
            X = self.get_expression_mat(split)
            uniq_label_strings, y = np.unique(self.splits[split]['labels_series'].values, return_inverse=True)
            siam_X, siam_y = siamese.create_flexible_data_pairs(X, y, self.get_true_ids(split), uniq_label_strings, args)
            print('Siamese X shape:')
            print(siam_X.shape)
            self.splits[split]['siam_X'] = [ siam_X[:, 0], siam_X[:, 1] ]
            self.splits[split]['siam_y'] = siam_y
            makedirs(cache_path)
            np.save(join(cache_path, "siam_X"), siam_X)
            np.save(join(cache_path, "siam_y"), siam_y)
    
    def create_siamese_data(self, args):
        for split in self.splits.keys():
            self._create_siamese_data_split(args, split)

class ExpressionSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, name, shuffle=True):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.name = name
        if shuffle:
            idx_array = np.arange(self.x.shape[0])
            np.random.shuffle(idx_array)
            self.x = self.x[idx_array]
            self.y = self.y[idx_array]

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print("ExpressionSequence {} idx={}".format(self.name, idx))
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
