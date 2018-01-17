# import pdb; pdb.set_trace()
import math
from os import makedirs
from os.path import join, exists

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import SGD
from keras.utils import Sequence
from keras.callbacks import Callback, EarlyStopping
from keras import backend as K
from theano import tensor as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from sklearn.utils import shuffle

from .data_container import DataContainer
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


class LossHistory(Callback):
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def plot(self):
        plt.figure()
        plt.plot(self.losses)
        plt.title('Train loss history, every update')
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.legend(['train'], loc='upper right')
        plt.savefig(join(self.out_dir, "batch_history.png"))
        plt.close()
        plt.figure()
        
    
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
            self.batch_count = 0
            self.batch_out = join(out_dir, "every_batch")
            if not exists(self.batch_out):
                makedirs(self.batch_out)
            self.batch_plot_files = []
        if self.on_epoch:
            self.epoch_out = join(out_dir, "every_epoch")
            if not exists(self.epoch_out):
                makedirs(self.epoch_out)
            self.epoch_plot_files = []

    def on_batch_end(self, batch, logs={}):
        if self.on_batch:
            self.batch_plot_files.append(self.plot(self.batch_count, self.batch_out))
            self.batch_count += 1
    
    def on_epoch_end(self, epoch, logs={}):
        if self.on_epoch:
            self.epoch_plot_files.append(self.plot(epoch, self.epoch_out))

    def plot(self, count, out_dir):
        embedding = self.embedding_model.predict(self.x)
        fig, ax = plt.subplots()
        ax.scatter(embedding[:,0], embedding[:,1], c=self.y)
        if self.fixed_axis_limits:
            ax.set_xlim(self.xlim[0], self.xlim[1])
            ax.set_ylim(self.ylim[0], self.ylim[1])
        else:
            self.fixed_axis_limits = True
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            self.xlim = ax.get_xlim()
            self.ylim = ax.get_ylim()
        filename = join(out_dir, "plot_" + str(count) + ".png")
        fig.savefig(filename)
        plt.close()
        return filename

    def animate(self):
        if self.on_batch:
            self.make_gif(self.batch_plot_files, join(self.out_dir, "batches.gif"))
        if self.on_epoch:
            self.make_gif(self.epoch_plot_files, join(self.out_dir, "epochs.gif"))

    def make_gif(self, image_files, out_file):
        images = []
        for filename in image_files:
            images.append(imageio.imread(filename))
        imageio.mimsave(out_file, images)


def base_net(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(1000, activation='tanh')(inputs)
    x = Dense(100, activation='tanh')(x)
    #x = Dense(2, activation='tanh')(x)
    x = Dense(output_dim, activation='softmax')(x)
    base_net = Model(name="base_net", inputs=inputs, outputs=x)
    return base_net

def get_triplet_batch_hard_loss(batch_size):
    def triplet_batch_hard_loss(y_true, y_pred):
        # y_pred is the embedding, y_true is the IDs (labels) of the samples (not 1-hot encoded)
        # They are mini-batched. If batch_size is B, and embedding dimension is D, shapes are:
        #   y_true: (B,)
        #   y_pred: (B,D)
        margin = 0.2
    
        # Get all-pairs distances
        y_true = K.sum(y_true, axis=1)
        diffs = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
        dist_mat = K.sqrt(K.sum(K.square(diffs), axis=-1) + K.epsilon())
        same_identity_mask = K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0))
        # TODO: make this backend-agnostic somehow
        negative_mask = T.bitwise_not(same_identity_mask)
        # XOR ensures that the same sample is paired with itself
        positive_mask = T.bitwise_xor(same_identity_mask, K.eye(batch_size, dtype='bool'))
        #print(K.int_shape(y_true))
        #print(K.int_shape(y_pred))

        #positive_mask = T.bitwise_xor(same_identity_mask, T.eye(K.int_shape(y_true)[0]))

        furthest_positive = K.max(dist_mat*positive_mask, axis=1)
        #closest_negative = K.min(dist_mat*negative_mask + np.inf*same_identity_mask, axis=1)
        closest_negative = K.min(dist_mat*negative_mask + 1e6*same_identity_mask, axis=1)

        loss = K.maximum(furthest_positive - closest_negative + margin, 0)
        return loss
    return triplet_batch_hard_loss

# def triplet_batch_hard_loss(y_true, y_pred):
#     # y_pred is the embedding, y_true is the IDs (labels) of the samples (not 1-hot encoded)
#     # They are mini-batched. If batch_size is B, and embedding dimension is D, shapes are:
#     #   y_true: (B,)
#     #   y_pred: (B,D)
#     margin = 0.2
    
#     # Get all-pairs distances
#     y_true = K.sum(y_true, axis=1)
#     diffs = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
#     dist_mat = K.sqrt(K.sum(K.square(diffs), axis=-1) + K.epsilon())
#     same_identity_mask = K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0))
#     # TODO: make this backend-agnostic somehow
#     negative_mask = T.bitwise_not(same_identity_mask)
#     # XOR ensures that the same sample is paired with itself
#     positive_mask = T.bitwise_xor(same_identity_mask, K.eye(72, dtype='bool'))
#     #print(K.int_shape(y_true))
#     #print(K.int_shape(y_pred))

#     #positive_mask = T.bitwise_xor(same_identity_mask, T.eye(K.int_shape(y_true)[0]))

#     furthest_positive = K.max(dist_mat*positive_mask, axis=1)
#     #closest_negative = K.min(dist_mat*negative_mask + np.inf*same_identity_mask, axis=1)
#     closest_negative = K.min(dist_mat*negative_mask + 1e6*same_identity_mask, axis=1)

#     loss = K.maximum(furthest_positive - closest_negative + margin, 0)
#     return loss

def get_frac_active_triplet_metric(batch_size):
    def frac_active_triplet_metric(y_true, y_pred):
        loss = get_triplet_batch_hard_loss(batch_size)(y_true, y_pred)
        num_active = K.sum(K.greater(loss, 1e-5))
        return num_active/batch_size
    return frac_active_triplet_metric
    

def get_triplet_net(base_net):
    embedding = base_net.layers[-2].output
    return Model(name="triplet_net", inputs=base_net.layers[0].input, outputs=embedding)
    

def main():
    # get a dataset
    dataset_file = '/home/aalavi/scrna_nn/data/mouse_data_20171220-154552/our_traindb_data.h5'
    data = DataContainer(dataset_file, False, True)
    X, y, _ = data.get_data()
    X, y = shuffle(X, y)
    print("All data shape: ", X.shape)

    input_dim = X.shape[1]
    output_dim = max(y) + 1
    
    base_model = base_net(input_dim, output_dim)
    triplet_model = get_triplet_net(base_model)
    print(triplet_model.summary())
    sgd = SGD(lr=0.01, nesterov=True)
    frac_active_triplet_metric = get_frac_active_triplet_metric(72)
    triplet_model.compile(loss=get_triplet_batch_hard_loss(72), metrics=[frac_active_triplet_metric], optimizer=sgd)
    print("Successfully compiled model")
    X_train = X[0:20480]
    y_train = y[0:20480]
    X_valid = X[20480:21760]
    y_valid = y[20480:21760]
    train_data = TripletSequence(X_train, y_train, 100, num_batches=1000)
    valid_data = TripletSequence(X_valid, y_valid, 100, num_batches=100)
    #train_data = TripletSequence(X_train, y_train, 100, num_batches=1)
    #valid_data = TripletSequence(X_valid, y_valid, 100, num_batches=1)
    #history = triplet_model.fit(X_train, y_train, batch_size=256, epochs=25, verbose=1, validation_data=(X_valid, y_valid))
    plot_samples = np.random.randint(0, 20480, 300)
    X_plot = X[plot_samples]
    y_plot = y[plot_samples]
    plotter = Plotter(triplet_model, X_plot, y_plot, '/scratch/aalavi/', on_batch=True, on_epoch=True)
    loss_hist = LossHistory('/scratch/aalavi/')
    # history = triplet_model.fit_generator(train_data, epochs=10, verbose=0, callbacks=[plotter], validation_data=valid_data)
    # plotter.animate()
    # history = triplet_model.fit_generator(train_data, epochs=30, verbose=1, callbacks=[loss_hist], validation_data=valid_data)
    # loss_hist.plot()
    early_stop = EarlyStopping(monitor='frac_active_triplet_metric', patience=10, verbose=1, mode='min')
    # early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    history = triplet_model.fit_generator(train_data, epochs=100, verbose=1, callbacks=[loss_hist, early_stop], validation_data=valid_data)
    loss_hist.plot()
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['frac_active_triplet_metric'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid', '% active triplets'], loc='upper left')
    plt.savefig(join('/scratch/aalavi/', 'epoch_history.png'))
    plt.close()
    plt.figure()
    plt.semilogy(history.history['frac_active_triplet_metric'])
    plt.title('Fraction of active triplets per epoch')
    plt.ylabel('% active triplets')
    plt.xlabel('epoch')
    plt.savefig(join('/scratch/aalavi/', 'frac_active_triplets.png'))
    plt.close()

if __name__ == "__main__":
    main()
