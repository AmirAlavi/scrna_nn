"""Cluster cells

Usage:
    cluster_cells.py <data> <model> (--view|--explore=<k_range>|--assign=<k>) [ --out=<path> ]

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn
seaborn.set()
import FisherExact
from scipy.stats import binom_test

from scrna_nn.data_container import DataContainer
from scrna_nn.reduce import _reduce_helper
from scrna_nn import util


def create_legend_markers(ordered_classes, color_map):
        legend_circles = []
        for cls in ordered_classes:
                color = color_map[cls]
                circ = mpatches.Circle((0,0), 1, fc=color)
                legend_circles.append(circ)
        return legend_circles
        

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

def plot_groups(X, name, working_dir, labels, config):
        colors = []
        for l in labels:
                for group, color in config['group_color_map'].items():
                        if group in l:
                                colors.append(color)
        circles = create_legend_markers(config['groups'], config['group_color_map'])
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.65)
        plt.title(name)
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.legend(circles, config['groups'], ncol=2, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.savefig(join(working_dir, name+"_groups.pdf"), bbox_inches="tight")
        plt.savefig(join(working_dir, name+"_groups.png"), bbox_inches="tight")
        plt.close()
        
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

def reduce_dimensions(args):
    reduced_by_model, _ = _reduce_helper(args['<model>'], args['<data>'])
    return reduced_by_model

def main(args):
        working_dir = util.create_working_directory(args['--out'], 'cluster_cells/')
        #model_name = basename(normpath(args['<model>']))
        #baseline_name = basename(normpath(args['<baseline>']))
        #raw_query, query_labels, raw_db, db_labels, db_cell_ids, query_data_container = load_data(args)
        reduced_data = reduce_dimensions(args)
        # 1, visualize the data using 2D PCA and 2D t-SNE
        visualize(data['original_query'], "original_query_data", working_dir, data['query_labels'], config)
        # 2, reduce dimensions of data using the trained models, visualize using 2D PCA and 2D t-sne
        query_model, db_model, query_baseline, db_baseline = reduce_dimensions(args)
        visualize(query_baseline, baseline_name+"_reduced", working_dir, data['query_labels'], config)
        visualize(query_model, model_name+"_reduced", working_dir, data['query_labels'], config)

        top_5_nearest_cells = defaultdict(list)
        
        nearest_dist_per_type_dict = defaultdict(lambda: defaultdict(list))
        top_5_types_dict = defaultdict(list)
        avg_nearest_5_distances_dict = defaultdict(list)
        classifications = defaultdict(list)
        overall_classifications = []
        
        dist_model = distance.cdist(query_model, db_model, metric='euclidean')
        for index, distances_to_query in enumerate(dist_model):
                query_label = data['query_labels'][index]
                sorted_distances_indices = np.argsort(distances_to_query)
                sorted_distances = distances_to_query[sorted_distances_indices]
                sorted_labels = data['db_labels'][sorted_distances_indices]
                sorted_cell_ids = data['db_cell_ids'][sorted_distances_indices]

                top_5_nearest_cells[data['query_datacontainer'].rpkm_df.index[index]] = sorted_cell_ids[:5]
                
                min_distances = nearest_dist_to_each_type(sorted_distances, sorted_labels)
                for db_type, min_dist in min_distances.items():
                        nearest_dist_per_type_dict[query_label][db_type].append(min_dist)

                top_5_types = sorted_labels[:5]
                top_5_types_dict[query_label].extend(top_5_types)

                avg_top_5_distances = np.mean(sorted_distances[:5])
                avg_nearest_5_distances_dict[query_label].append(avg_top_5_distances)

                classified_label = classify(sorted_labels[:100])
                overall_classifications.append(classified_label)
                classifications[query_label].append(classified_label)
                
        make_nearest_dist_per_db_type_plots(nearest_dist_per_type_dict, working_dir)
        make_top_5_labels_plots(top_5_types_dict, working_dir)
        make_5_nearest_distances_plots(avg_nearest_5_distances_dict, working_dir)
        make_classification_histograms(classifications, working_dir, config)

if __name__ == "__main__":
        args = docopt(__doc__)
        print(args)
        main(args)
