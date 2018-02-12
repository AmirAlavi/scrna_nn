import math

import numpy as np
from keras.utils import Sequence

from . import util


class TripletSequence(Sequence):
    def __init__(self, x_set, y_set, y_dim, ids_per_batch=18, samples_per_id=4, num_batches=1000):
        # for each id in y_set, get a list of indices into x_set for that id
        self.x_set = x_set
        self.y_set = y_set
        self.y_dim = y_dim
        self.ids_per_batch = ids_per_batch
        self.samples_per_id = samples_per_id
        self.num_batches = num_batches

        self.batch_size = ids_per_batch * samples_per_id
        self.id_dict = util.build_indices_master_list(x_set, y_set)
        self.possible_ids = np.array(list(self.id_dict.keys()))
        print("possible_ids.shape: ", self.possible_ids.shape)
        
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.x_set.shape[1]), dtype='float32')
        y = np.zeros((self.batch_size, self.y_dim), dtype='float32')
        if self.ids_per_batch <= len(self.possible_ids):
            selected_ids = np.random.choice(self.possible_ids, self.ids_per_batch, replace=False)
        else:
            selected_ids = np.random.choice(self.possible_ids, self.ids_per_batch, replace=True)
        for i, cur_id in enumerate(selected_ids):
            cur_id_samples = np.array(self.id_dict[cur_id])
            count = cur_id_samples.shape[0]
            num_to_take = math.ceil(self.samples_per_id/count) * count
            padded_list_indexer = np.mod(np.arange(num_to_take), count)
            selection = np.random.choice(padded_list_indexer, self.samples_per_id, replace=False)
            selected = cur_id_samples[selection]
            X[i*self.samples_per_id:(i+1)*self.samples_per_id, :] = self.x_set[selected]
            y[i*self.samples_per_id:(i+1)*self.samples_per_id, 0] = cur_id
        return X, y        
