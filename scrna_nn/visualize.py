import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .data_container import DataContainer
from .util import create_working_directory
from .train import build_indices_master_list


def visualize(args):
    working_dir_path = create_working_directory(args['--out'], "visualizations/")
    # Load the reduced data
    data = DataContainer(args['<reduced_data_file>'])
    X = data.get_expression_mat()
    print(X.shape)
    pca = PCA(n_components=2)
    print("Fitting PCA model...")
    X = pca.fit_transform(X)
    print("Fitted.")
    y = data.get_labels()
    #labels_and_counts = zip(np.unique(y, return_counts=True))
    #labels_and_counts = sorted(labels_and_counts, key=lambda x: x[1], reverse=True)

    # Pick enough different colors
    colormap = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_colors = int(args['--ntypes'])
    # Pre-set pyplot's color cycler
    ax.set_color_cycle([colormap(1.*i/num_colors) for i in range(num_colors)])

    indices_lists = build_indices_master_list(X, y) # dict<label, list of indices of samples>
    labels_and_counts = [(label, len(samples)) for label, samples in indices_lists.items()]
    labels_and_counts.sort(key=lambda x: x[1], reverse=True)
    
    max_cells = int(args['--nsamples'])
    for i in range(int(args['--ntypes'])):
        # Select at most max_cells of each type and plot them
        print(labels_and_counts[i])
        cur_label = labels_and_counts[i][0]
        cur_X = X[indices_lists[cur_label]][:max_cells]
        ax.scatter(cur_X[:,0], cur_X[:,1], label=cur_label, alpha=0.3) 
    lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, fancybox=True, shadow=True)
    plt.title(args['--title'])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    print("Saving...")
    fig.savefig(join(working_dir_path, "embedding.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    np.save(join(working_dir_path, "X.npy"), X)
    np.save(join(working_dir_path, "y.npy"), y)
    with open(join(working_dir_path, "model.pickle"), 'wb') as f:
        pickle.dump(pca, f)
    print("done")
