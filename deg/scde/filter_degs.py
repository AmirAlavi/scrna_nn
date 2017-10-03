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

def get_deg_files(folder):
    node_names = []
    file_paths = []
    for file in listdir(folder):
        if file.endswith("_meta.csv"):
            node_name = file.split('_')[0]
            node_names.append(node_name)
            file_paths.append(join(folder, file))
    return node_names, file_paths

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
    node_names, file_paths = get_deg_files(args['<input_folder>'])
    split_degs(node_names, file_paths, out_path)
