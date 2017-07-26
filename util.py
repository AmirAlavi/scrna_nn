import time
from os.path import exists, join
from os import makedirs

class ScrnaException(Exception):
    pass

def create_working_directory(out_path, parent, suffix=""):
    if out_path == 'None':
        time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
        out_path = join(parent ,time_str + "_" + suffix)
    if not exists(out_path):
        makedirs(out_path)
    return out_path
