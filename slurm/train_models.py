"""Train various scrna-nn models

Usage:
    train_models.py <partition> <mem> <cpus> <email> <model_types> <input_data> <output_root> [--scratch --unsup_pt_models=<folder>] [<shared_options>]
    train_models.py (-h | --help)

Options:
    -s --scratch                            Use scratch disc
    -u <folder> --unsup_pt_models=<folder>  Folder that contains unsupervised pretrained models
    -h --help                               Show this screen
"""
#import pdb; pdb.set_trace()
import sys
import os
import string
import subprocess

from docopt import docopt


class SafeDict(dict):
    """Allows for string formatting with unused keyword arguments
    """

    def __missing__(self, key):
        return '{' + key + '}'

SCRATCH_PREFIX = '/scratch/aalavi/'

PCA_DIMS = ['1136', '500', '200', '100', '50']
DENSE_LAYERS = [['1136'], ['1136', '100'], ['1136', '500'], ['1136', '500', '100'], ['1136', '500', '100', '50']]
PPITF_LAYERS = [layers[1:] for layers in DENSE_LAYERS]
FLATGO_LAYERS = [[], ['100'], ['200'], ['200', '100'], ['200', '50']]
COMBINED_MODELS_LAYERS = [[], ['100']]

PT_DENSE_LAYERS = [['1136', '100'], ['1136', '500', '100']]
PT_PPITF_LAYERS = [layers[1:] for layers in PT_DENSE_LAYERS]
PT_FLATGO_LAYERS = [['100'], ['200', '100']]

COMMON_COMMAND = "scrna-nn train {model_specific_opts} --out={out_path} --data={in_data} {shared_opts}"


def get_output_path(args, model_name):
    out_path = os.path.join(args['<output_root>'], model_name)
    args['model_output_locations'].append(out_path)
    if args['--scratch']:
        out_path = os.path.join(SCRATCH_PREFIX, out_path)
    return out_path

def add_cp_from_scratch(command, scratch_source_location, args):
    return command + " && cp -r {} {}".format(scratch_source_location, args['<output_root>'])

def parse_args():
    args = docopt(__doc__)
    if args['<shared_options>'] is not None:
        args['<shared_options>'] = " ".join(args['<shared_options>'].split('*')[1:])
    if 'pretrained' in args['<model_types>'] and args['--unsup_pt_models'] is None:
        raise ValueError("Must specify --unusp_pt_models if training pretrained model types!")
    args['<model_types>'] = args['<model_types>'].split(',')
    print(args)
    return args

def get_base_command(template, model_name, args):
    out_path = get_output_path(args, model_name)
    command = string.Formatter().vformat(template, (),
                                         SafeDict(out_path=out_path,
                                                  in_data=args['<input_data>'],
                                                  shared_opts=args['<shared_options>']))
    if args['--scratch']:
             command = add_cp_from_scratch(command, out_path, args)
    return command

def pca(args):
    pca_template = string.Formatter().vformat(COMMON_COMMAND, (), SafeDict(model_specific_opts="--pca={n_components}"))
    for components in PCA_DIMS:
        name = "pca_{}".format(components)
        command = get_base_command(pca_template, name, args)
        command = string.Formatter().vformat(command, (), SafeDict(n_components=components))
        args['commands_list'].append(command)

def nn_command_construction_helper(args, nn_type, layer_list, name_prefix, base_name, other_model_opts=""):
    model_specific_opts = string.Formatter().vformat("--nn={nn_type} {hidden_sizes} {other_opts}", (), SafeDict(nn_type=nn_type))
    nn_template = string.Formatter().vformat(COMMON_COMMAND, (), SafeDict(model_specific_opts=model_specific_opts))
    for hiddens in layer_list:
        name = "_".join([base_name] + hiddens)
        prefixed_name = name
        if len(name_prefix) > 0:
            prefixed_name = name_prefix + name
        command = get_base_command(nn_template, prefixed_name, args)
        unsup_pt_arg = ""
        if 'pt' in name_prefix:
            pt_model_path = os.path.join(args['--unsup_pt_models'], name)
            unsup_pt_arg = " --unsup_pt={}".format(pt_model_path)
        command = string.Formatter().vformat(command, (), SafeDict(hidden_sizes=" ".join(hiddens), other_opts=other_model_opts+unsup_pt_arg))
        args['commands_list'].append(command)
    
def dense(args, layers_list, name_prefix, other_opts=""):
    nn_command_construction_helper(args, "dense", layers_list, name_prefix, "dense", other_opts)
    
def ppitf(args, layers_list, name_prefix, other_opts=""):
    other_opts += " --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100"
    nn_command_construction_helper(args, "sparse", layers_list, name_prefix, "ppitf_1036.100", other_opts)

def flatGO(args, layers_list, name_prefix, other_opts=""):
    other_opts += " --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100"
    nn_command_construction_helper(args, "sparse", layers_list, name_prefix, "flatGO_300.100", other_opts)
    
def hierarchicalGO(args, name_prefix, other_opts=""):
    other_opts += " --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31"
    nn_command_construction_helper(args, "GO", [[]], name_prefix, "hierarchicalGO", other_opts)

def combined_flatGO_ppitf(args, layers_list, name_prefix, other_opts=""):
    other_opts += " --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt"
    nn_command_construction_helper(args, "flatGO_ppitf", layers_list, name_prefix, "comb_flatGO_ppitf", other_opts)

def combined_flatGO_ppitf_dense(args, layers_list, name_prefix, other_opts=""):
    other_opts += " --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=100"
    nn_command_construction_helper(args, "flatGO_ppitf", layers_list, name_prefix, "comb_flatGO_ppitf_dense", other_opts)

def create_neural_net_train_commands(args, name_prefix, unsupervised_pretraining=False, other_opts=""):
    if unsupervised_pretraining:
        dense(args, PT_DENSE_LAYERS, name_prefix, other_opts)
        ppitf(args, PT_PPITF_LAYERS, name_prefix, other_opts)
        flatGO(args, PT_FLATGO_LAYERS, name_prefix, other_opts)
        hierarchicalGO(args, name_prefix, other_opts)
    else:
        dense(args, DENSE_LAYERS, name_prefix, other_opts)
        ppitf(args, PPITF_LAYERS, name_prefix, other_opts)
        flatGO(args, FLATGO_LAYERS, name_prefix, other_opts)
        hierarchicalGO(args, name_prefix, other_opts)
        combined_flatGO_ppitf(args, COMBINED_MODELS_LAYERS, name_prefix, other_opts)
        combined_flatGO_ppitf_dense(args, COMBINED_MODELS_LAYERS, name_prefix, other_opts)

def glup(args):
    original_output_root = args['<output_root>']
    args['<output_root>'] = os.path.join(args['<output_root>'], 'glup')
    create_neural_net_train_commands(args, name_prefix="", unsupervised_pretraining=True, other_opts="--ae --layerwise_pt")
    args['<output_root>'] = original_output_root

def write_list_to_file(my_list, filename):
    with open(filename, 'w') as f:
        for elt in my_list:
            f.write(elt + "\n")

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args['<output_root>']):
        os.makedirs(args['<output_root>'])
    # Add shared data structures to the args dictionary for convenience and conciseness
    args['commands_list'] = []
    args['model_output_locations'] = []
    for model_type in args['<model_types>']:
        if model_type == "pca":
            pca(args)
        elif model_type == "non-siamese":
            create_neural_net_train_commands(args, name_prefix="")
        elif model_type == "siamese":
            create_neural_net_train_commands(args, name_prefix="siam_", other_opts="--siamese")
        elif model_type == "triplet":
            create_neural_net_train_commands(args, name_prefix="triplet_", other_opts="--triplet")
        elif model_type == "pretrained":
            create_neural_net_train_commands(args, unsupervised_pretraining=True, name_prefix="pt_")
        elif model_type == "siamese-pretrained":
            create_neural_net_train_commands(args, name_prefix="pt_siam_", unsupervised_pretraining=True, other_opts="--siamese")
        elif model_type == "triplet-pretrained":
            create_neural_net_train_commands(args, name_prefix="pt_triplet_", unsupervised_pretraining=True, other_opts="--triplet")
        elif model_type == "GLUP": # Greedy Layerwise Unsupervised Pretraining
            glup(args)
        else:
            raise ValueError("Not a valid model type: {}".format(model_type))
    train_commands_filename = os.path.join(args['<output_root>'], 'train_commands.list')
    write_list_to_file(args['commands_list'], train_commands_filename)
    write_list_to_file(args['model_output_locations'], os.path.join(args['<output_root>'], 'model_output_locations.list'))
    num_jobs = len(args['commands_list']) - 1
    gres = "--gres=gpu:1" if args['<partition>'] == 'gpu' else ""
    job_name = "train-{}".format("-".join(args['<model_types>']))
    slurm_out = os.path.join(args['<output_root>'], 'scrna_train_array_%A_%a.out')
    slurm_err = os.path.join(args['<output_root>'], 'scrna_train_array_%A_%a.err')
    sbatch_cmd = "sbatch --job-name={job_name} -p {partition} {gres} -n 1 -c {num_cpus} --mem-per-cpu={mem}  --array=0-{num_jobs} --mail-user {email} --mail-type=FAIL --output {slurm_out} --error {slurm_err} slurm/train_model_arr.sh {train_cmds_file}"
    sbatch_cmd = sbatch_cmd.format(job_name=job_name,
                                   partition=args['<partition>'],
                                   gres=gres,
                                   num_cpus=args['<cpus>'],
                                   mem=args['<mem>'],
                                   num_jobs=num_jobs,
                                   email=args['<email>'],
                                   slurm_out=slurm_out,
                                   slurm_err=slurm_err,
                                   train_cmds_file=train_commands_filename
    )
    print(sbatch_cmd)
    subprocess.run(sbatch_cmd.split())
# TODO: create a command to do greedy layerwise pretraining
