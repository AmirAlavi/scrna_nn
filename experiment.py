# import pdb; pdb.set_trace()
from os.path import join, exists, basename, normpath
from os import makedirs
import time
from collections import defaultdict
import string
import sys
import subprocess


DEFAULT_WORKING_DIR_ROOT='experiments'
DEFAULT_MODELS_FILE='experiment_models.list'
REDUCE_COMMAND_TEMPLATE="""python scrna.py reduce {trained_nn_folder} \
--data=data/integrate_imputing_dataset_kNN10_simgene_T.txt --out_folder={output_folder}"""

RETRIEVAL_COMMAND_TEMPLATE="""python scrna.py retrieval {reduced_data_folder} \
--out_folder={output_folder}"""

SLURM_TRANSFORM_COMMAND="""sbatch --array=0-{num_jobs} --mail-user {email} \
--output {out_folder}/scrna_transform_array_%A_%a.out
--error {err_folder}/scrna_transform_array_%A_%a.err slurm_transform_array.sh"""

SLURM_RETRIEVAL_COMMAND="""sbatch --array=0-{num_jobs} --mail-user {email} \
--output {out_folder}/scrna_retrieval_array_%A_%a.out
--error {err_folder}/scrna_retrieval_array_%A_%a.err -d afterok:{depends} slurm_retrieval_array.sh"""

class SafeDict(dict):
    """Allows for string formatting with unused keyword arguments
    """
    def __missing__(self, key):
        return '{' + key + '}'


class Experiment(object):
    def __init__(self, working_dir_path=None):
        if not working_dir_path:
            # Automatically create a unique working directory for the experiment
            time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
            working_dir_path = join(DEFAULT_WORKING_DIR_ROOT, time_str)
        makedirs(working_dir_path)
        self.working_dir_path = working_dir_path
        
    def prepare(self, models_file=DEFAULT_MODELS_FILE):
        # Prep Transform commands
        print("Preparing Transform commands...")
        with open(models_file) as f:
            model_folders = f.readlines()
        model_folders = [s.strip() for s in model_folders]
        transform_commands = {}
        transform_data_folders = {}
        for model_folder in model_folders:
            model_name = basename(normpath(model_folder))
            reduced_data_folder = join(self.working_dir_path, "data_transformed_by_" + model_name)
            transform_data_folders[model_name] = reduced_data_folder
            transform_commands[model_name] = string.Formatter().vformat(REDUCE_COMMAND_TEMPLATE, (), SafeDict(trained_nn_folder=model_folder, output_folder=reduced_data_folder))
        with open('transform_commands.list', 'w') as f:
            for value in transform_commands.values():
                f.write(value + '\n')
        self.transform_commands = transform_commands
        self.transform_data_folders = transform_data_folders
        # Prep Retrieval commands
        print("Preparing Retrieval commands...")
        retrieval_commands = {}
        for model_name, transformed_data_folder in transform_data_folders.items():
            retrieval_result_folder = join(self.working_dir_path, "retrieval_results/" + model_name)
            retrieval_commands[model_name] = string.Formatter().vformat(RETRIEVAL_COMMAND_TEMPLATE, (), SafeDict(reduced_data_folder=transformed_data_folder, output_folder=retrieval_result_folder))
        with open('retrieval_commands.list', 'w') as f:
            for value in retrieval_commands.values():
                f.write(value + '\n')
        self.retrieval_commands = retrieval_commands
        print("Preparation complete, commands constructed.")

    def run(self, email_addr):
        # First transform the data
        slurm_transform_out_folder = join(self.working_dir_path, "slurm_transform_out")
        makedirs(slurm_transform_out_folder)
        num_jobs = len(self.transform_commands)
        transform_cmd = SLURM_TRANSFORM_COMMAND.format(num_jobs=str(num_jobs-1), email=email_addr, out_folder=slurm_transform_out_folder, err_folder=slurm_transform_out_folder)
        print("Running slurm array job to reduce dimensions using models...")
        result = subprocess.run(transform_cmd.split(), stdout=subprocess.PIPE)
        transform_job_id = int(result.stdout.decode("utf-8").strip().split()[-1])
        print("Slurm array job submitted, id: ", transform_job_id)
        # Then run retrieval (after transformation completes)
        slurm_retrieval_out_folder = join(self.working_dir_path, "slurm_retrieval_out")
        makedirs(slurm_retrieval_out_folder)
        num_jobs = len(self.retrieval_commands)
        retrieval_cmd = SLURM_RETRIEVAL_COMMAND.format(num_jobs=str(num_jobs-1), email=email_addr, out_folder=slurm_retrieval_out_folder, err_folder=slurm_retrieval_out_folder, depends=transform_job_id)
        print("Running slurm array job to conduct retrieval test using each model...")
        result = subprocess.run(retrieval_cmd.split(), stdout=subprocess.PIPE)
        retrieval_job_id = int(result.stdout.decode("utf-8").strip().split()[-1])
        print("Slurm array job submitted, id: ", retrieval_job_id)

if __name__ == '__main__':
    email_addr = sys.argv[1]
    exp = Experiment()
    exp.prepare()
    exp.run(email_addr)
