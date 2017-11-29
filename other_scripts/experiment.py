"""Retrieval Experiment Runner

Usage:
    experiment.py <model_list_file> <query_file> <db_file> <partition> <email_address>

Options:
    -h --help  Show this screen.
"""
import pickle
import string
import subprocess
import time
from os import makedirs
# import pdb; pdb.set_trace()
from os.path import join, basename, normpath, realpath, dirname

from docopt import docopt

DEFAULT_WORKING_DIR_ROOT = 'experiments'
REDUCE_COMMAND_TEMPLATE = """scrna-nn reduce {trained_nn_folder} \
--data={data_file} --out={output_file} --save_meta"""

RETRIEVAL_COMMAND_TEMPLATE = """scrna-nn retrieval {reduced_query_file} {reduced_db_file} \
--out={output_folder} --dist_mat_file=dist_mat_by_strings.p"""

SLURM_TRANSFORM_COMMAND = """sbatch -p {partition} --array=0-{num_jobs} --mail-user {email} \
--output {out_folder}/scrna_transform_array_%A_%a.out
--error {err_folder}/scrna_transform_array_%A_%a.err {slurm_script}"""

SLURM_RETRIEVAL_COMMAND = """sbatch -p {partition} --array=0-{num_jobs} --mail-user {email} \
--output {out_folder}/scrna_retrieval_array_%A_%a.out
--error {err_folder}/scrna_retrieval_array_%A_%a.err -d afterok:{depends} {slurm_script}"""

def get_slurm_transform_script_path():
    return join(dirname(realpath(__file__)), '../slurm/slurm_transform_array.sh')


def get_slurm_retrieval_script_path():
    return join(dirname(realpath(__file__)), '../slurm/slurm_retrieval_array.sh')


class SafeDict(dict):
    """Allows for string formatting with unused keyword arguments
    """

    def __missing__(self, key):
        return '{' + key + '}'


def write_out_command_dict(cmd_dict, path):
    with open(path, 'w') as f:
        for value in cmd_dict.values():
            # TODO: hack
            if isinstance(value, tuple):
                f.write(value[0] + '\n')
                f.write(value[1] + '\n')
            else:
                f.write(value + '\n')


class Experiment(object):
    def __init__(self, working_dir_path=None):
        if not working_dir_path:
            # Automatically create a unique working directory for the experiment
            time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
            working_dir_path = join(DEFAULT_WORKING_DIR_ROOT, time_str)
        makedirs(working_dir_path)
        self.working_dir_path = working_dir_path

    def prepare(self, models_file, query_file, db_file):
        """
        Args:
            models_file: path to a file which contains, on each line, the path
                         to the folder containing a trained neural network
                         model.
        """
        # Prep Transform commands
        print("Preparing Transform commands...")
        with open(models_file) as f:
            model_folders = f.readlines()
        model_folders = [s.strip() for s in model_folders]
        transform_commands = {}
        transform_data_folders = {}
        for model_folder in model_folders:
            model_name = basename(normpath(model_folder))
            # path to output location, where the transformed data will be written to
            reduced_data_folder = join(self.working_dir_path, "data_transformed_by_" + model_name)
            reduced_query_file = join(reduced_data_folder, "reduced_query.h5")
            reduced_db_file = join(reduced_data_folder, "reduced_db.h5")
            transform_data_folders[model_name] = reduced_data_folder
            transform_query = string.Formatter().vformat(REDUCE_COMMAND_TEMPLATE, (),
                                                         SafeDict(trained_nn_folder=model_folder, data_file=query_file,
                                                                  output_file=reduced_query_file))
            transform_db = string.Formatter().vformat(REDUCE_COMMAND_TEMPLATE, (),
                                                      SafeDict(trained_nn_folder=model_folder, data_file=db_file,
                                                               output_file=reduced_db_file))
            transform_commands[model_name] = (transform_query, transform_db)
        # write each of the command lines for transformation to a file, to be consumed by the slurm jobs
        write_out_command_dict(transform_commands, 'transform_commands.list')
        self.transform_commands = transform_commands
        self.transform_data_folders = transform_data_folders
        # Prep Retrieval commands
        print("Preparing Retrieval commands...")
        retrieval_dir = join(self.working_dir_path, "retrieval_results")
        makedirs(retrieval_dir)
        retrieval_commands = {}
        retrieval_result_folders = {}
        for model_name, transformed_data_folder in transform_data_folders.items():
            # path to output location, where the retrieval test results will be written to
            retrieval_result_folder = join(retrieval_dir, model_name)
            transformed_query = join(transformed_data_folder, "reduced_query.h5")
            transformed_db = join(transformed_data_folder, "reduced_db.h5")
            retrieval_commands[model_name] = string.Formatter().vformat(RETRIEVAL_COMMAND_TEMPLATE, (),
                                                                        SafeDict(reduced_query_file=transformed_query,
                                                                                 reduced_db_file=transformed_db,
                                                                                 output_folder=retrieval_result_folder))
            retrieval_result_folders[model_name] = retrieval_result_folder
        # Also compare with using raw, undreduced data
        orig_model_name = "original_data"
        orig_retrieval_result_folder = join(retrieval_dir, orig_model_name)
        retrieval_commands[orig_model_name] = string.Formatter().vformat(RETRIEVAL_COMMAND_TEMPLATE, (),
                                                                         SafeDict(reduced_query_file=query_file,
                                                                                  reduced_db_file=db_file,
                                                                                  output_folder=orig_retrieval_result_folder))
        retrieval_result_folders[orig_model_name] = orig_retrieval_result_folder
        # write each of the command lines for retrieval testing to a file, to be consumed by the slurm jobs
        write_out_command_dict(retrieval_commands, 'retrieval_commands.list')
        self.retrieval_commands = retrieval_commands
        self.retrieval_dir = retrieval_dir
        self.retrieval_result_folders = retrieval_result_folders
        print("Preparation complete, commands constructed.")

    def run(self, partition, email_addr):
        # First transform the data
        slurm_transform_out_folder = join(self.working_dir_path, "slurm_transform_out")
        makedirs(slurm_transform_out_folder)
        number_jobs = len(self.transform_commands) * 2
        slurm_trans_path = get_slurm_transform_script_path()
        transform_cmd = SLURM_TRANSFORM_COMMAND.format(partition=partition, num_jobs=str(number_jobs - 1),
                                                       email=email_addr,
                                                       out_folder=slurm_transform_out_folder,
                                                       err_folder=slurm_transform_out_folder,
                                                       slurm_script=slurm_trans_path)
        print("Running slurm array job to reduce dimensions using models...")
        result = subprocess.run(transform_cmd.split(), stdout=subprocess.PIPE)
        transform_job_id = int(result.stdout.decode("utf-8").strip().split()[-1])
        print("Slurm array job submitted, id: ", transform_job_id)
        # Then run retrieval (after transformation completes)
        slurm_retrieval_out_folder = join(self.working_dir_path, "slurm_retrieval_out")
        makedirs(slurm_retrieval_out_folder)
        number_jobs = len(self.retrieval_commands)
        slurm_retr_path = get_slurm_retrieval_script_path()
        retrieval_cmd = SLURM_RETRIEVAL_COMMAND.format(partition=partition, num_jobs=str(number_jobs - 1),
                                                       email=email_addr,
                                                       out_folder=slurm_retrieval_out_folder,
                                                       err_folder=slurm_retrieval_out_folder, depends=transform_job_id,
                                                       slurm_script=slurm_retr_path)
        print("Running slurm array job to conduct retrieval test using each model...")
        result = subprocess.run(retrieval_cmd.split(), stdout=subprocess.PIPE)
        retrieval_job_id = int(result.stdout.decode("utf-8").strip().split()[-1])
        print("Slurm array job submitted, id: ", retrieval_job_id)
        # Must wait for retrieval jobs to finish in order to use their results
        print("Waiting for retrieval jobs to finish...")
        wait_cmd = "srun -J completion -d afterok:{depends} --mail-type END,FAIL --mail-user {email} -p {partition} echo '(done waiting)'"
        subprocess.run(wait_cmd.format(partition=partition, depends=retrieval_job_id, email=email_addr).split())

    def write_out_table(self, compiled_results, file_prefix, metric_header, metric_key, mean_metric_key):
        # get list of cell types in the results
        any_model_results = next(iter(compiled_results.values()))
        cell_types = sorted(any_model_results["cell_types"].keys())
        with open(join(self.working_dir_path, file_prefix + '_results_table.csv'), 'w', newline='') as csv_file:
            csv_file.write("# in Query")
            for label in cell_types:
                csv_file.write("," + str(any_model_results["cell_types"][label]["#_in_query"]))
            csv_file.write("\n# in Database")
            for label in cell_types:
                csv_file.write("," + str(any_model_results["cell_types"][label]["#_in_DB"]))
            csv_file.write("\nModel")
            for label in cell_types:
                csv_file.write("," + label)

            csv_file.write("," + metric_header + ",Weighted " + metric_header + "\n")
            for model_name, results in compiled_results.items():
                csv_file.write(model_name + ",")
                for label in cell_types:
                    csv_file.write(str(results["cell_types"][label][metric_key]) + ",")
                csv_file.write(str(results[mean_metric_key]) + "," + str(results["weighted_"+mean_metric_key]) + "\n")

    def compile_results(self):
        """Compiles results into various tables:
        - overall table - contains all of the raw data in a single table

        root_folder: the path to the folder that contains a folder for each model
        Returns: overall results table
        """
        compiled_results = {}  # <model_name:dict>
        for model_name, results_folder in self.retrieval_result_folders.items():
            # Iterate through models
            current_model = {}  # <cell_type:score>
            print(model_name)
            results_file = join(results_folder, 'retrieval_results_d.pickle')
            with open(results_file, 'rb') as f:
                compiled_results[model_name] = pickle.load(f)
        self.write_out_table(compiled_results, "map", "Average MAP", "Mean_Average_Precision", "average_map")
        self.write_out_table(compiled_results, "mafp", "Average MAFP", "Mean_Average_Flex_Precision", "average_mafp")
        self.write_out_table(compiled_results, "mafp2", "Average MAFP2", "Mean_Average_Flex_Precision2", "average_mafp2")
        self.write_out_table(compiled_results, "mac", "Average MAC", "Mean_Average_Accuracy", "average_mac")
        self.write_out_table(compiled_results, "macq", "Average MACQ", "Mean_Average_Accuracy_of_top_quarter", "average_macq")

if __name__ == '__main__':
    args = docopt(__doc__, version='experiment 0.1')
    exp = Experiment()
    exp.prepare(args['<model_list_file>'], args['<query_file>'], args['<db_file>'])
    exp.run(args['<partition>'], args['<email_address>'])
    exp.compile_results()
