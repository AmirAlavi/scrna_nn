import time
from os.path import exists, join
from os import makedirs
from collections import defaultdict

class ScrnaException(Exception):
    pass

def create_working_directory(out_path, parent, suffix=""):
    if out_path == 'None':
        time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
        out_path = join(parent ,time_str + "_" + suffix)
    if not exists(out_path):
        makedirs(out_path)
    return out_path

def build_indices_master_list(X, y):
    '''Builds a mapping of label (encoded as int) to a list
    of indices of training examples that have that label.
    '''
    indices_lists = defaultdict(list) # dictionary of lists
    print(X.shape[0], "examples in dataset")
    for sample_idx in range(X.shape[0]):
        indices_lists[y[sample_idx]].append(sample_idx)
    return indices_lists
