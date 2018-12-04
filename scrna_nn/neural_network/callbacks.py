import pickle
import math
from os import makedirs
from os.path import join, exists

import imageio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from keras.callbacks import Callback
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from ..data_manipulation.data_container import DataContainer


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

    def on_train_end(self, logs={}):
        if self.on_batch:
            self.make_gif(self.batch_plot_files, join(self.out_dir, "batches.gif"))
        if self.on_epoch:
            self.make_gif(self.epoch_plot_files, join(self.out_dir, "epochs.gif"))        

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

    def make_gif(self, image_files, out_file):
        images = []
        for filename in image_files:
            images.append(imageio.imread(filename))
        imageio.mimsave(out_file, images)

class TSNEPlotter(Callback):
    def __init__(self,
                 embedding_model,
                 data,
                 out_dir,
                 interval=1,
                 sample_normalize=False,
                 feature_normalize=False,
                 feature_mean=None,
                 feature_std=None,
                 minmax_normalize=False,
                 minmax_scaler=None):                 
        self.embedding_model = embedding_model
        self.data = data
        self.interval = interval
        with open(join(self.data, 'color_map.pickle'), 'rb') as f:
            color_map = pickle.load(f)
        train_data = DataContainer(join(self.data, 'train_data.h5'),
                                   sample_normalize=sample_normalize,
                                   feature_normalize=feature_normalize,
                                   feature_mean=feature_mean,
                                   feature_std=feature_std,
                                   minmax_normalize=minmax_normalize,
                                   minmax_scaler=minmax_scaler)
        self.X_train = train_data.get_expression_mat()
        self.y_train = train_data.get_labels()
        self.colors_train = [color_map[y] for y in self.y_train]
        valid_data = DataContainer(join(self.data, 'valid_data.h5'),
                                   sample_normalize=sample_normalize,
                                   feature_normalize=feature_normalize,
                                   feature_mean=feature_mean,
                                   feature_std=feature_std,
                                   minmax_normalize=minmax_normalize,
                                   minmax_scaler=minmax_scaler)
        self.X_valid = valid_data.get_expression_mat()
        self.y_valid = valid_data.get_labels()
        self.colors_valid = [color_map[y] for y in self.y_valid]
        # build legend
        self.legend_elements = []
        for label, color in color_map.items():
            circ = Line2D([0], [0], marker='o', color='w',
                          label=label, markerfacecolor=color,
                          markersize=15)
            self.legend_elements.append(circ)
        self.out_dir = out_dir
        if not exists(self.out_dir):
            makedirs(self.out_dir)
        self.train_out_dir = join(self.out_dir, 'train')
        if not exists(self.train_out_dir):
            makedirs(self.train_out_dir)
        self.valid_out_dir = join(self.out_dir, 'valid')
        if not exists(self.valid_out_dir):
            makedirs(self.valid_out_dir)
        self.combined_out_dir = join(self.out_dir, 'combined')
        if not exists(self.combined_out_dir):
            makedirs(self.combined_out_dir)
        self.train_plot_files = []
        self.valid_plot_files = []
        self.combined_plot_files = []

    def on_train_begin(self, logs={}):
        self.plot('init')
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            self.plot(epoch)

    def on_train_end(self, logs={}):
        self.make_gif(self.train_plot_files, join(self.out_dir, "train.gif"))
        self.make_gif(self.valid_plot_files, join(self.out_dir, "valid.gif"))
        self.make_gif(self.combined_plot_files, join(self.out_dir, "combined.gif"))

    def plot(self, name):
        X_train_embed = self.embedding_model.predict(self.X_train)
        X_valid_embed = self.embedding_model.predict(self.X_valid)
        tSNE_all = TSNE().fit_transform(np.concatenate((X_train_embed, X_valid_embed), axis=0))
        # Train only
        fig, ax = plt.subplots()
        train_end = X_train_embed.shape[0]
        ax.scatter(tSNE_all[:train_end,0], tSNE_all[:train_end,1], c=self.colors_train, marker='o', alpha=0.5)
        lgd = ax.legend(handles=self.legend_elements, bbox_to_anchor=(1.05, 1), loc=2, fancybox=True, shadow=True)
        filename = join(self.train_out_dir, "plot_" + str(name) + ".png")
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        self.train_plot_files.append(filename)
        # Valid only
        fig, ax = plt.subplots()
        ax.scatter(tSNE_all[train_end:,0], tSNE_all[train_end:,1], c=self.colors_valid, marker='x')
        lgd = ax.legend(handles=self.legend_elements, bbox_to_anchor=(1.05, 1), loc=2, fancybox=True, shadow=True)
        filename = join(self.valid_out_dir, "plot_" + str(name) + ".png")
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        self.valid_plot_files.append(filename)
        # Combined
        fig, ax = plt.subplots()
        ax.scatter(tSNE_all[:train_end,0], tSNE_all[:train_end,1], c=self.colors_train, marker='o', alpha=0.5)
        ax.scatter(tSNE_all[train_end:,0], tSNE_all[train_end:,1], c=self.colors_valid, marker='x')
        lgd = ax.legend(handles=self.legend_elements, bbox_to_anchor=(1.05, 1), loc=2, fancybox=True, shadow=True)
        filename = join(self.combined_out_dir, "plot_" + str(name) + ".png")
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        self.combined_plot_files.append(filename)

    def make_gif(self, image_files, out_file):
        images = []
        for filename in image_files:
            images.append(imageio.imread(filename))
        imageio.mimsave(out_file, images)

class PCAPlotter(Callback):
    def __init__(self,
                 pca_model,
                 embedding_model,
                 data,
                 out_dir,
                 interval=1,
                 sample_normalize=False,
                 feature_normalize=False,
                 feature_mean=None,
                 feature_std=None,
                 minmax_normalize=False,
                 minmax_scaler=None):
        with open(pca_model, 'rb') as f:
            self.pca_model = pickle.load(f)
        self.embedding_model = embedding_model
        self.data = data
        self.interval = interval
        with open(join(self.data, 'color_map.pickle'), 'rb') as f:
            color_map = pickle.load(f)
        train_data = DataContainer(join(self.data, 'train_data.h5'),
                                   sample_normalize=sample_normalize,
                                   feature_normalize=feature_normalize,
                                   feature_mean=feature_mean,
                                   feature_std=feature_std,
                                   minmax_normalize=minmax_normalize,
                                   minmax_scaler=minmax_scaler)
        self.X_train = train_data.get_expression_mat()
        self.y_train = train_data.get_labels()
        self.colors_train = [color_map[y] for y in self.y_train]
        valid_data = DataContainer(join(self.data, 'valid_data.h5'),
                                   sample_normalize=sample_normalize,
                                   feature_normalize=feature_normalize,
                                   feature_mean=feature_mean,
                                   feature_std=feature_std,
                                   minmax_normalize=minmax_normalize,
                                   minmax_scaler=minmax_scaler)
        self.X_valid = valid_data.get_expression_mat()
        self.y_valid = valid_data.get_labels()
        self.colors_valid = [color_map[y] for y in self.y_valid]
        # build legend
        self.legend_elements = []
        for label, color in color_map.items():
            circ = Line2D([0], [0], marker='o', color='w',
                          label=label, markerfacecolor=color,
                          markersize=15)
            self.legend_elements.append(circ)
        self.out_dir = out_dir
        if not exists(self.out_dir):
            makedirs(self.out_dir)
        self.train_out_dir = join(self.out_dir, 'train')
        if not exists(self.train_out_dir):
            makedirs(self.train_out_dir)
        self.valid_out_dir = join(self.out_dir, 'valid')
        if not exists(self.valid_out_dir):
            makedirs(self.valid_out_dir)
        self.combined_out_dir = join(self.out_dir, 'combined')
        if not exists(self.combined_out_dir):
            makedirs(self.combined_out_dir)
        self.train_plot_files = []
        self.valid_plot_files = []
        self.combined_plot_files = []

    def on_train_begin(self, logs={}):
        self.plot('init')
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            self.plot(epoch)

    def on_train_end(self, logs={}):
        self.make_gif(self.train_plot_files, join(self.out_dir, "train.gif"))
        self.make_gif(self.valid_plot_files, join(self.out_dir, "valid.gif"))
        self.make_gif(self.combined_plot_files, join(self.out_dir, "combined.gif"))

    def plot(self, name):
        X_train_embed = self.embedding_model.predict(self.X_train)
        X_valid_embed = self.embedding_model.predict(self.X_valid)
        pca_all = self.pca_model.transform(np.concatenate((X_train_embed, X_valid_embed), axis=0))
        # Train only
        fig, ax = plt.subplots()
        train_end = X_train_embed.shape[0]
        ax.scatter(pca_all[:train_end,0], pca_all[:train_end,1], c=self.colors_train, marker='o', alpha=0.5)
        lgd = ax.legend(handles=self.legend_elements, bbox_to_anchor=(1.05, 1), loc=2, fancybox=True, shadow=True)
        filename = join(self.train_out_dir, "plot_" + str(name) + ".png")
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        self.train_plot_files.append(filename)
        # Valid only
        fig, ax = plt.subplots()
        ax.scatter(pca_all[train_end:,0], pca_all[train_end:,1], c=self.colors_valid, marker='x')
        lgd = ax.legend(handles=self.legend_elements, bbox_to_anchor=(1.05, 1), loc=2, fancybox=True, shadow=True)
        filename = join(self.valid_out_dir, "plot_" + str(name) + ".png")
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        self.valid_plot_files.append(filename)
        # Combined
        fig, ax = plt.subplots()
        ax.scatter(pca_all[:train_end,0], pca_all[:train_end,1], c=self.colors_train, marker='o', alpha=0.5)
        ax.scatter(pca_all[train_end:,0], pca_all[train_end:,1], c=self.colors_valid, marker='x')
        lgd = ax.legend(handles=self.legend_elements, bbox_to_anchor=(1.05, 1), loc=2, fancybox=True, shadow=True)
        filename = join(self.combined_out_dir, "plot_" + str(name) + ".png")
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        self.combined_plot_files.append(filename)

    def make_gif(self, image_files, out_file):
        images = []
        for filename in image_files:
            images.append(imageio.imread(filename))
        imageio.mimsave(out_file, images)


class LossHistory(Callback):
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.epoch_losses = []
        self.epoch_val_losses = []

    def on_train_end(self, logs={}):
        plt.figure()
        plt.plot(self.batch_losses)
        plt.title('Train loss history, every update')
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.savefig(join(self.out_dir, "batch_history.png"))
        plt.close()
        # also save info as text
        with open(join(self.out_dir, "batch_history.txt"), 'w') as f:
            f.write('{:6} {:4}\n'.format("Update", "Loss"))
            for epoch, loss in enumerate(self.batch_losses):
                f.write('{:6d} {:012.8f}\n'.format(epoch+1, float(loss)))
        plt.figure()
        plt.plot(self.epoch_losses)
        plt.plot(self.epoch_val_losses)
        plt.title('Loss history, every epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(join(self.out_dir, 'epoch_history.png'))
        plt.close()
        # also save info as text
        with open(join(self.out_dir, "epoch_history.txt"), 'w') as f:
            f.write('{:5} {:12} {:12}\n'.format("Epoch", "Loss", "Val_Loss"))
            for epoch in range(len(self.epoch_losses)):
                f.write('{:5d} {:012.8f} {:012.8f}\n'.format(epoch+1, self.epoch_losses[epoch], self.epoch_val_losses[epoch]))

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        self.epoch_losses.append(logs.get('loss'))
        self.epoch_val_losses.append(logs.get('val_loss'))


class StepLRHistory(Callback):
    """Adapted from Suki Lau's blog post:
           'Learning Rate Schedules and Adaptive Learning Rate Methods for Deep Learning'
           https://medium.com/towards-data-science/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    """
    def __init__(self, initial_lr, epochs_drop, out_dir):
        self.initial_lr = initial_lr
        self.epochs_drop = epochs_drop
        self.out_dir = out_dir
        self.step_decay_fcn = self.get_step_decay_fcn()

    def get_step_decay_fcn(self):
        def step_decay(epoch):
            drop = 0.5
            lr = self.initial_lr * math.pow(drop, math.floor((epoch)/float(self.epochs_drop)))
            return lr
        return step_decay

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.lr = []

    def on_train_end(self, logs={}):
        plt.figure()
        epochs = np.arange(1, len(self.losses)+1)
        plt.plot(epochs, self.losses)
        plt.plot(epochs, self.val_losses)
        plt.plot(epochs, self.lr, marker='o', linestyle='None')
        plt.title('Learning Rate & Loss History')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid', 'lr'], loc='upper right')
        plt.savefig(join(self.out_dir, "lr_history.png"))
        plt.close()
    
    def on_epoch_end(self, batch, logs={}):
        zero_indexed_epoch_num = len(self.losses)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.lr.append(self.step_decay_fcn(zero_indexed_epoch_num))

class EarlyStoppingAtValue(Callback):
    def __init__(self, monitor='val_loss', target=1e-5, verbose=0):
        super(EarlyStoppingAtValue, self).__init__()
        
        self.monitor = monitor
        self.target = target
        self.verbose = verbose
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            print('Warning: Early stopping conditioned on metric `%s` '
                  'which is not available.' % self.monitor)
            return
        if np.less_equal(current, self.target):
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
