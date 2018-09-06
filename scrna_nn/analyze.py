from os.path import join

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import util
from .data_manipulation.data_container import DataContainer
#from .reduce import _reduce_helper
from . import reduce


def visualize(data, working_dir):
    original_dims = data.shape[1]
    pca = PCA(n_components=2)
    pca_reduced = pca.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pca_reduced[:,0], pca_reduced[:,1])
    plt.title("{:d} reduced by PCA".format(original_dims))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    fig.savefig(join(working_dir, "pca_{:d}.png".format(original_dims)))
    plt.close()

    tsne = TSNE(n_components=2)
    tsne_reduced = tsne.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(tsne_reduced[:,0], tsne_reduced[:,1])
    plt.title("{:d} reduced by t-SNE".format(original_dims))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    fig.savefig(join(working_dir, "tsne_{:d}.png".format(original_dims)))
    plt.close()


# analyze <model> <raw_data> <raw_database> --out=...
def analyze(args):
    working_dir_path = util.create_working_directory(args.out, "analyses/")
    query_data = DataContainer(args.query)
    #database_data = DataContainer(args['<database>'])
    query = query_data.get_expression_mat()
    #database = database_data.get_expression_mat()

    # First, visualize the unreduced query data
    visualize(query, working_dir_path)

    # Then embed it and visualize it
    query_reduced, _ = reduce._reduce_helper(args.trained_model_folder, args.query)
    visualize(query_reduced, working_dir_path)

    

    
