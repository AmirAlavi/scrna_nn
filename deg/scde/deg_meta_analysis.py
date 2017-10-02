"""Analayse the results of DEG.

Usage:
    deg_analysis.py <input_folder> <output_folder>

Options:
    -h --help                Show this screen.
    --version                Show version.
"""
# In each csv file, first col is col 1, col 4 is fold change, col 6 is p-val
# import pdb; pdb.set_trace()
from os import listdir, makedirs
from os.path import exists, join
from collections import defaultdict

from docopt import docopt
import pandas as pd

def get_matching_filenames(folder, suffix=".csv"):
    files = []
    for file in listdir(folder):
        if file.endswith(suffix):
            files.append(join(folder, file))
    return files

def get_filenames_by_groups(folder):
    filenames = {
        "all": get_matching_filenames(folder, "_all.csv"),
        "up": get_matching_filenames(folder, "_up.csv"),
        "down": get_matching_filenames(folder, "_down.csv")
    }
    return filenames

def get_files_for_nodes(folder):
    node_files = defaultdict(list)
    for file in listdir(folder):
        if file.endswith("all.csv"):
            node = file.split('_')[0]
            node_files[node].append(join(folder, file))
    return node_files

def combine_de_results(node_files, log_handle, out_path):
    for node, files in node_files.items():
        if len(files) == 0:
            log_handle.write("Node " + node + " only had one experiment.\n")
        else:
            max_p_vals = []
            # For all genes, go through all experiments and get the max adj p-values as it's new p-value
            # Then readjust FDR and take genes with new adj p-value below a threshold
            
def get_intersecting_degs(node_files, log_handle, out_path):
    for node, files in node_files.items():
        if len(files) == 0:
            log_handle.write("Node " + node + " only had one experiment.\n")
        else:
            # Get the intersection of the significant genes from each experiment
            deg_lists = []
            for file in files:
                df = pd.read_csv(file)
                deg_lists.append(df['EntrezID'].tolist())
            intersect = set(deg_lists[0]).intersection(*deg_lists)
            log_handle.write("Node " + node + " had " + str(len(files)) + " experiments, and intersection of " + str(len(intersect)) + " genes.\n")
            with open(join(out_path, node + "_intersection.txt"), 'w') as f:
                for gene in intersect:
                    f.write(str(gene) + "\n")

if __name__ == "__main__":
    args = docopt(__doc__, version="deg_analysis 0.1")
    
    out_path = args['<output_folder>']
    if not exists(out_path):
        makedirs(out_path)
    # Write stats about the analysis to a log
    log = open(join(out_path, 'log.txt'), 'w')
        
    node_files = get_files_for_nodes(args['<input_folder>'])
    get_intersecting_degs(node_files, log, out_path)
    
    log.close()
    
