import math
from os import makedirs
from os.path import join

import matplotlib
import numpy as np
from keras.callbacks import Callback
from keras.utils import Sequence

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .train import build_indices_master_list

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
        self.id_dict = build_indices_master_list(x_set, y_set)
        self.possible_ids = np.array(id_dict.keys())
        
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.x_set.shape[1]), dtype='float32')
        y = np.zeros((self.batch_size,, self.y_dim), dtype='float23')
        selected_ids = np.random.choice(self.possible_ids, self.ids_per_batch, replace=False)
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


def triplet_batch_hard_loss(y_true, y_pred):
    # y_pred is the embedding, y_true is the IDs (labels) of the samples (not 1-hot encoded,
    # just zero-padded as a hack to make Keras happy about dimensions matching).
    # They are mini-batched. If batch_size is B, and embedding dimension is D, shapes are:
    #   y_true: (B,D)
    #   y_pred: (B,D)
    margin = 0.2
    # Get all-pairs distances
    y_true = K.sum(y_true, axis=1) # Hack. Gets rid of the zero padding we had to add
    diffs = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
    dist_mat = K.sqrt(K.sum(K.square(diffs), axis=-1) + K.epsilon())
    same_identity_mask = K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0))
    # TODO: make this backend-agnostic somehow
    negative_mask = T.bitwise_not(same_identity_mask)
    # XOR ensures that the same sample is paired with itself
    positive_mask = T.bitwise_xor(same_identity_mask, K.eye(256, dtype='bool'))

    furthest_positive = K.max(dist_mat*positive_mask, axis=1)
    closest_negative = K.min(dist_mat*negative_mask + 1e6*same_identity_mask, axis=1)

    loss = K.maximum(furthest_positive - closest_negative + margin, 0)
    return loss

class Plotter(Callback):
    def __init__(self, embedding_model, x, y, out_dir, on_batch=False, on_epoch=False):
        self.embedding_model = embedding_model
        self.x = x
        self.y = y
        self.out_dir = out_dir
        self.on_batch = on_batch
        self.on_epoch = on_epoch
        
        self.fixed_axis_limits = False
        if self.on_batch:
            self.batch_out = join(out_dir, "every_batch")
            makedirs(self.batch_out)
        if self.on_epoch:
            self.epoch_out = join(out_dir, "every_epoch")
            makedirs(self.epoch_out)

    def on_batch_end(self, batch, logs={}):
        if self.on_batch:
            self.plot(batch, self.batch_out)
    
    def on_epoch_end(self, epoch, logs={}):
        if self.on_epoch:
            self.plot(batch, self.epoch_out)

    def plot(self, count, out_dir):
        embedding = self.embedding_model.predict(self.x)
        fig, ax = plt.subplots
        ax.scatter(embedding[:,0], embedding[:,1], self.y)
        if self.fixed_axis_limits:
            ax.set_xlim(self.xlim[0], self.xlim[1])
            ax.set_ylim(self.ylim[0], self.ylim[1])
        else:
            self.fixed_axis_limits = True
            self.xlim = ax.get_xlim()
            self.ylim = ax.get_ylim()
        fig.savefig(join(out_dir, "plot_" + str(count) + ".png"))
        plt.close()
        
        
def train_triplet_model(model, data_container, working_dir_path, args):
    
