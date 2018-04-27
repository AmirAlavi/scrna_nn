"""Comparative Analysis

Usage:
    comparative_analysis.py <query_data> <db_data> <model> <baseline> <config_json> [ --out=<path> ]

Options:
    -h --help     Show this screen.
    --out=<path>  Path to save output to. 'None' means a time-stamped folder
                  will automatically be created. [default: None]
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

def load_data(args):
        query_data = DataContainer(args['<query_data>'])
        db_data = DataContainer(args['<db_data>'])
        raw_query = query_data.get_expression_mat()
        raw_db = db_data.get_expression_mat()
        query_labels = query_data.get_labels()
        db_labels = db_data.get_labels()
        db_cell_ids = db_data.get_cell_ids()
        data = {
                'query_datacontainer': query_data,
                'db_datacontainer': db_data,
                'original_query': raw_query,
                'original_db': raw_db,
                'query_labels': query_labels,
                'db_labels': db_labels,
                'db_cell_ids': db_cell_ids,
                }
        return data

def reduce_dimensions(args):
        query_reduced_by_model, _ = _reduce_helper(args['<model>'], args['<query_data>'])
        db_reduced_by_model, _ = _reduce_helper(args['<model>'], args['<db_data>'])
        query_reduced_by_baseline, _ = _reduce_helper(args['<baseline>'], args['<query_data>'])
        db_reduced_by_baseline, _ = _reduce_helper(args['<baseline>'], args['<db_data>'])
        return query_reduced_by_model, db_reduced_by_model, query_reduced_by_baseline, db_reduced_by_baseline

def nearest_dist_to_each_type(distances, labels):
        min_distances = {}
        for dist, label in zip(distances, labels):
                if label not in min_distances:
                        min_distances[label] = dist
        return min_distances

def make_nearest_dist_per_db_type_plots(d, working_dir):
        working_dir = join(working_dir, "A_nearest_dist_per_db_type")
        if not exists(working_dir):
                makedirs(working_dir)
        for query_type, nearest_to_db_types_d in d.items():
                avgs_labels = []
                avgs_values = []
                for db_type, nearest_distances in nearest_to_db_types_d.items():
                        avgs_labels.append(db_type)
                        avgs_values.append(np.mean(nearest_distances))
                avgs_labels = np.array(avgs_labels)
                avgs_values = np.array(avgs_values)
                sort_idx = np.argsort(avgs_values)
                avgs_labels = avgs_labels[sort_idx]
                avgs_values = avgs_values[sort_idx]

                plt.figure()
                plt.bar(np.arange(1, len(avgs_labels)+1), avgs_values, tick_label=avgs_labels)
                plt.xticks(rotation='vertical', fontsize=8)
                plt.title("Avg of nearest distances to each DB type")
                plt.savefig(join(working_dir, query_type+".pdf"), bbox_inches="tight")
                plt.savefig(join(working_dir, query_type+".png"), bbox_inches="tight")
                plt.close()
                

def make_top_5_labels_plots(d, working_dir):
        working_dir = join(working_dir, "B_top_5_label_frequencies")
        if not exists(working_dir):
                makedirs(working_dir)
        for query_type, top_fives in d.items():
                db_types, counts = np.unique(top_fives, return_counts=True)
                sort_idx = np.argsort(counts)[::-1]
                db_types = db_types[sort_idx]
                counts = counts[sort_idx]
                plt.figure()
                plt.bar(np.arange(1, len(db_types)+1), counts, tick_label=db_types)
                plt.xticks(rotation='vertical', fontsize=8)
                plt.title("# of times each DB type appeared in top 5 results")
                plt.savefig(join(working_dir, query_type+".pdf"), bbox_inches="tight")
                plt.savefig(join(working_dir, query_type+".png"), bbox_inches="tight")
                plt.close()

def make_5_nearest_distances_plots(d, working_dir):
        working_dir = join(working_dir, "C_nearest_5_distances")
        if not exists(working_dir):
                makedirs(working_dir)
        for query_type, avg_nearest_distances in d.items():
                plt.figure()
                plt.hist(avg_nearest_distances, bins='auto')
                plt.title("Hist, avg top 5 nearest distances, mean={}".format(np.mean(avg_nearest_distances)))
                plt.savefig(join(working_dir, query_type+".pdf"), bbox_inches="tight")
                plt.savefig(join(working_dir, query_type+".png"), bbox_inches="tight")
                plt.close()

def plot_classification_histograms_single_group(db_types, counts, query_type, working_dir):
        stripped_db_types = [' '.join(name.split()[1:]) for name in db_types]
        plt.figure()
        plt.bar(np.arange(1, len(stripped_db_types)+1), counts, tick_label=stripped_db_types)
        plt.xticks(rotation='vertical', fontsize=8)
        plt.title("# of query cells that are classified as each cell type")
        plt.savefig(join(working_dir, query_type+".pdf"), bbox_inches="tight")
        plt.savefig(join(working_dir, query_type+".png"), bbox_inches="tight")
        plt.close()
        
def plot_classification_histograms_two_group(group_data, working_dir, name, config, ignore_zero=False):
        # merge db_types
        merged_db_types = set()
        for clfs in group_data.values():
                merged_db_types.update(clfs)
        merged_db_types = np.array(list(merged_db_types))
        normalized_data = defaultdict(list)
        for group, clfs in group_data.items():
                counts = Counter(clfs)
                for label in merged_db_types:
                        if label in counts:
                                normalized_data[group].append(counts[label])
                        else:
                                normalized_data[group].append(0)
                normalized_data[group] = np.array(normalized_data[group])
        if ignore_zero:
                keep = []
                for i in range(len(merged_db_types)):
                        zero = False
                        for clfs in normalized_data.values():
                                if clfs[i] == 0:
                                        zero = True
                        if not zero:
                                keep.append(i)
                merged_db_types = merged_db_types[keep]
                for group in normalized_data.keys():
                        normalized_data[group] = normalized_data[group][keep]
                                        
        # Normalize
        for group in normalized_data.keys():
                normalized_data[group] = normalized_data[group] / np.sum(normalized_data[group])
        bar_width = 0.35
        x_locations = np.arange(1, len(merged_db_types)+1)
        # Sort the bars by a group, any group
        sort_idx = np.argsort(normalized_data[config['groups'][0]])[::-1]
        merged_db_types = merged_db_types[sort_idx]
        merged_db_types = [' '.join(name.split()[1:]) for name in merged_db_types]
        for group, fracs in normalized_data.items():
                normalized_data[group] = normalized_data[group][sort_idx]
        plt.figure()
        rects = []
        for i, group in enumerate(config['groups']):
                rects.append(plt.bar(x_locations + bar_width*i, normalized_data[group], bar_width, color=config['group_color_map'][group], label=config['group_names'][group]))
        plt.xticks(x_locations + bar_width / 2, merged_db_types, rotation='vertical', fontsize=8)
        plt.ylabel('Fraction of query cells')
        plt.legend()
        plt.title("Portion of query cells classified as each cell type")
        # add p-vals
        for rect1, rect2 in zip(rects[0], rects[1]):
                height1 = rect1.get_height()
                height2 = rect2.get_height()
                count1 = math.floor(1e6*height1)
                count2 = math.floor(1e6*height2)
                pval = binom_test(count1, count1+count2, p=0.5)
                plt.gca().text(rect1.get_x(), 1.05*max(height1,height2), '{:.2e}'.format(pval), ha='left', va='bottom', rotation='vertical', fontsize=6)
        plt.savefig(join(working_dir, name + ".pdf"), bbox_inches="tight")
        plt.savefig(join(working_dir, name + ".png"), bbox_inches="tight")
        plt.close()

def make_classification_histograms_for_groups(classifications, working_dir, config):
        group_clfs = defaultdict(list)
        for query_type, clfs in classifications.items():
                for group in config['groups']:
                        if group in query_type:
                                group_clfs[group].extend(clfs)
        if len(config['groups']) == 2:
                plot_classification_histograms_two_group(group_clfs, working_dir, "groups", config, ignore_zero=False)
                plot_classification_histograms_two_group(group_clfs, working_dir, "groups_nonzero", config, ignore_zero=True)

def tmp_make_classification_histograms_for_groups(classifications, working_dir, config):
        # outer loop over time points:
        time_points = ["3m", "3m_1w", "3m_2w", "4m_2w"]
        for t in time_points:
                group_clfs = defaultdict(list)
                for query_type, clfs in classifications.items():
                        for group in config['groups']:
                                if query_type == group + "_" + t:
                                        group_clfs[group].extend(clfs)
                plot_classification_histograms_two_group(group_clfs, working_dir, t + "_groups", config, ignore_zero=False)
                plot_classification_histograms_two_group(group_clfs, working_dir, t + "_groups_nonzero", config, ignore_zero=True)
        
def make_classification_histograms(classifications, working_dir, config):
        working_dir = join(working_dir, "D_classification_histograms")
        if not exists(working_dir):
                makedirs(working_dir)
        # Make overall classification histogram
        overall_classifications = []
        for clfs in classifications.values():
                overall_classifications.extend(clfs)
        db_types, counts = np.unique(overall_classifications, return_counts=True)
        sort_idx = np.argsort(counts)[::-1]
        db_types = db_types[sort_idx]
        counts = counts[sort_idx]
        plot_classification_histograms_single_group(db_types, counts, 'overall', working_dir)
        # Make classification histograms for each query label
        for query_type, clfs in classifications.items():
                db_types, counts = np.unique(clfs, return_counts=True)
                sort_idx = np.argsort(counts)[::-1]
                db_types = db_types[sort_idx]
                counts = counts[sort_idx]
                plot_classification_histograms_single_group(db_types, counts, query_type, working_dir)

        if 'groups' in config:
                make_classification_histograms_for_groups(classifications, working_dir, config)
                # TEMPORARY HACK, only for mouse-brain data
                # Make classification plots for each time point between the two groups
                #tmp_make_classification_histograms_for_groups(classifications, working_dir, config)
                

def classify(sorted_neighbors):
        # Adapted from https://stackoverflow.com/a/1520716
        sorted_neighbors = sorted_neighbors.tolist()
        return max(g(sorted(sorted_neighbors)), key=lambda xv:(len(list(xv[1])),-sorted_neighbors.index(xv[0])))[0]
        
def load_config(args):
        with open(args['<config_json>'], 'r') as f:
                return json.load(f)

#def visualize():
#        pass

#def retrieval():
        
        

def main(args):
        config = load_config(args)
        working_dir = util.create_working_directory(args['--out'], 'comparative_analysis/')
        model_name = basename(normpath(args['<model>']))
        baseline_name = basename(normpath(args['<baseline>']))
        #raw_query, query_labels, raw_db, db_labels, db_cell_ids, query_data_container = load_data(args)
        data = load_data(args)
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

        top_fibroblast = []
        
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
                if 'fibroblast' in classified_label:
                        for i in range(100):
                                if 'fibroblast' in sorted_labels[i]:
                                        top_fibroblast.append(sorted_cell_ids[i])
                overall_classifications.append(classified_label)
                classifications[query_label].append(classified_label)
                
        make_nearest_dist_per_db_type_plots(nearest_dist_per_type_dict, working_dir)
        make_top_5_labels_plots(top_5_types_dict, working_dir)
        make_5_nearest_distances_plots(avg_nearest_5_distances_dict, working_dir)
        make_classification_histograms(classifications, working_dir, config)
        top_fibroblast = sorted(set(top_fibroblast))
        for f in top_fibroblast:
                print(f)
        with open('fibroblast_hits.txt', 'w') as f:
                for cell in top_fibroblast:
                        f.write("{}\n".format(cell))

if __name__ == "__main__":
        args = docopt(__doc__)
        print(args)
        main(args)
