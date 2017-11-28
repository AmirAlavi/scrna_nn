"""Analyze the Cell Ontology graphical structure

Usage:
    ontology_analysis_new.py <ontology_file> <output_file>
    ontology_analysis_new.py (-h | --help)
    ontology_analysis_new.py --version
"""
import pickle
from collections import defaultdict

import networkx as nx
from docopt import docopt
import numpy as np

import ontology

if __name__ == "__main__":
    args = docopt(__doc__, version='ontology_analysis_new 0.1')
    with open(args['<ontology_file>'], 'rb') as f:
        ont = pickle.load(f)
    undirected_graph = ont.dag.to_undirected()
    comps = list(nx.connected_component_subgraphs(undirected_graph))
    print("Number of connected components in the undirected graph: ", len(comps))
    num_nodes_list = [nx.number_of_nodes(g) for g in comps]
    largest_idx = np.argmax(num_nodes_list)
    # Number of nodes in largest is currently 5015
    print("Connected component with most nodes has ", num_nodes_list[largest_idx], " nodes")
    # Diameter of the largest one is currently 19
    #print("Diameter of this component is: ", nx.diameter(comps[largest_idx]))

    # Create distance matrix. Keyed by strings to be lightweight (rather than Term objects)
    lengths = nx.shortest_path_length(undirected_graph)
    print("Calculated shortest paths")
    dist_mat_by_strings = defaultdict(dict)
    out_file = open(args['<output_file>'], "w")
    for node1, dist_dict in lengths.items():
        for node2, dist in dist_dict.items():
            name1 = str(node1)
            name2 = str(node2)
            dist_mat_by_strings[name1][name2] = dist
            out_file.write(name1 + "\t" + name2 + "\t" + str(dist) + "\n")
    #with open(args['<output_file>'], 'wb') as f:
        #pickle.dump(dist_mat_by_strings, f)
    out_file.close()
    print("done")
    
