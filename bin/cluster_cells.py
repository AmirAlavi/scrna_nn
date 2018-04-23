"""Cluster cells

Usage:
    cluster_cells.py <data> <model> (--view|--explore=<k_range>|--assign=<k>) [--prefix=<prefix> --out=<path>]

Options:
    -h --help                         Show this screen.
    -o <path> --out=<path>            Path to save output to. 'None' means a time-stamped folder
                                      will automatically be created. [default: None]
    -v --view                         View a pre-labeled set of points in reduced dimensions.
    -e <k_range> --explore=<k_range>  Explore different clusterings (variouse values of k for k-means.
                                      k_range must be a tuple of the form (min, max). Parenthesis and comma
                                      are required.
    -a <k> --assign=<k>               Perform clustering with k-means with specified number of clusters k,
                                      and save these cluster annotations with each sample.
    -p <prefix> --prefix=<prefix>     String prefix to prepend to all cells in their labels and filenames.
"""
# import pdb; pdb.set_trace()
from os.path import join, basename, normpath, exists
from os import makedirs, remove
from collections import defaultdict, Counter
import json
from itertools import groupby as g
import math

import pandas as pd
import numpy as np
from scipy.spatial import distance
from docopt import docopt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
#import seaborn
#seaborn.set()
import FisherExact
from scipy.stats import binom_test

from scrna_nn.data_container import DataContainer
from scrna_nn.reduce import _reduce_helper
from scrna_nn import util
        

def plot(X, name, working_dir, labels, config):
        legend_circles = create_legend_markers(config['classes'], config['color_map'])
        # set up colors
        colors = [config['color_map'][label] for label in labels]
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.65)
        plt.title(name)
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.legend(legend_circles, config['classes'], ncol=2, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.savefig(join(working_dir, name+".pdf"), bbox_inches="tight")
        plt.savefig(join(working_dir, name+".png"), bbox_inches="tight")
        plt.close()

def plot_clustering(X, name, working_dir, labels, title=None):
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.7)
        if title is not None:
                plt.title(title)
        else:
                plt.title(name)
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.savefig(join(working_dir, name+".pdf"), bbox_inches="tight")
        plt.savefig(join(working_dir, name+".png"), bbox_inches="tight")
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, alpha=0.7)
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.set_zlabel('component 3')
        if title is not None:
                ax.set_title(title)
        else:
                ax.set_title(name)
        plt.savefig(join(working_dir, name+"_3d.pdf"), bbox_inches="tight")
        plt.savefig(join(working_dir, name+"_3d.png"), bbox_inches="tight")
        
        
def visualize(data, name, working_dir, labels, config):
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(data)
        plot(x_pca, name+"_PCA", working_dir, labels, config)
        tsne = TSNE(n_components=2)
        x_tsne = tsne.fit_transform(data)
        plot(x_tsne, name+"_TSNE", working_dir, labels, config)
        if 'groups' in config:
                plot_groups(x_pca, name+"_PCA", working_dir, labels, config)
                plot_groups(x_tsne, name+"_TSNE", working_dir, labels, config)


def view(args, data):
        pass

def explore(args, data, working_dir):
        k_range = args['--explore'].split(',')
        min_k, max_k = int(k_range[0]), int(k_range[1])
        #seaborn.set_palette("Set1", n_colors=max_k)
        scores = []
        for i in range(min_k, max_k):
                mkb = MiniBatchKMeans(n_clusters=i)
                mkb.fit(data)
                scores.append(mkb.score(data))
                y = mkb.predict(data)
                pca = PCA(n_components=3)
                x_pca = pca.fit_transform(data)
                plot_clustering(x_pca, "explore_{}_means".format(i), working_dir, y, "{}-means clustering of embedding + PCA".format(i))
        plt.figure()
        plt.plot(np.arange(min_k, max_k), scores)
        plt.title("Objective score vs k")
        plt.xlabel("k")
        plt.ylabel("-Objective score")
        plt.savefig(join(working_dir, "explore_scores.pdf"), bbox_inches="tight")
        plt.savefig(join(working_dir, "explore_scores.png"), bbox_inches="tight")
        plt.close()

def assign(args, data):
        k = int(args['--assign'])
        mkb = MiniBatchKMeans(n_clusters=k)
        mkb.fit(data)
        clusters = mkb.predict(data)
        labels = ['cluster_{}'.format(c) for c in clusters]
        # Get original DataFrame:
        store = pd.HDFStore(args['<data>'])
        df = store['rpkm']
        store.close()
        # Write out to an h5 file:
        filename = args['<data>'].split('.')
        filename = '.'.join(filename[:-1]) + "_labeled.hdf5"
        store = pd.HDFStore(filename)
        store['rpkm'] = df
        store['labels'] = pd.Series(data=labels, index=df.index)
        store.close()
        
def reduce_dimensions(args):
    reduced_by_model, _ = _reduce_helper(args['<model>'], args['<data>'])
    return reduced_by_model

def main(args):
        working_dir = util.create_working_directory(args['--out'], 'cluster_cells/')
        embedded_data = reduce_dimensions(args)

        if args['--view']:
                pass
        elif args['--explore']:
                explore(args, embedded_data, working_dir)
        elif args['--assign']:
                assign(args, embedded_data)

if __name__ == "__main__":
        args = docopt(__doc__)
        print(args)
        main(args)
