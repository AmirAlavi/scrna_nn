"""Filter and split the results of DEG meta analysis

Usage:
    filter_degs.py <input_folder> <output_folder>

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

def get_deg_files(folder):
    node_names = []
    file_paths = []
    for file in listdir(folder):
        if file.endswith("_meta.csv"):
            node_name = file.split('_')[0]
            node_names.append(node_name)
            file_paths.append(join(folder, file))
    return node_names, file_paths

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

def split_degs(node_names, node_files, out_path):
    for node_name, path in zip(node_names, node_files):
        df = pd.read_csv(path)
        up_df = df.loc[df['Avg_log2_fold_change'] > 0]
        dn_df = df.loc[df['Avg_log2_fold_change'] < 0]
        dn_df = dn_df.copy() # Hack to get rid of a warning
        dn_df.loc[:,'Avg_log2_fold_change'] = dn_df.loc[:,'Avg_log2_fold_change'].abs()
        up_df = up_df.sort_values(by=['Max_adj_p_value', 'Avg_log2_fold_change'], ascending=[True, False])
        dn_df = dn_df.sort_values(by=['Max_adj_p_value', 'Avg_log2_fold_change'], ascending=[True, False])
        up_df = up_df.head(100)
        dn_df = dn_df.head(100)
        up_df.to_csv(join(out_path, node_name + "_up.csv"))
        dn_df.to_csv(join(out_path, node_name + "_dn.csv"))

if __name__ == "__main__":
    args = docopt(__doc__, version="deg_analysis 0.1")
    
    out_path = args['<output_folder>']
    if not exists(out_path):
        makedirs(out_path)
    # Write stats about the analysis to a log
    #log = open(join(out_path, 'log.txt'), 'w')
    node_names, file_paths = get_deg_files(args['<input_folder>'])
    #node_files = get_files_for_nodes(args['<input_folder>'])
    split_degs(node_names, file_paths, out_path)
    
    #log.close()
    
