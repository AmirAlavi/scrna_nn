from os.path import join
import json
from collections import defaultdict

import numpy as np
from scipy.spatial import distance

from data_container import DataContainer
import common


def average_precision(target, retrieved_list):
    total = 0
    correct = 0
    avg_precision = 0
    for r in retrieved_list:
        total += 1
        if r == target:
            correct += 1
            avg_precision += correct/float(total)
    if correct > 0:
        avg_precision /= float(correct)
    return avg_precision

def retrieval_test(args):
    if args['--unreduced']:
        # Don't do anything to the data, use it raw
        model_type = 'original'
        data_file = args['<reduced_data_folder>']
    else:
        training_args_path = join(args['<reduced_data_folder>'], "training_command_line_args.json")
        with open(training_args_path, 'r') as fp:
            training_args = json.load(fp)
        model_type = training_args['--nn'] if training_args['--nn'] is not None else "pca"
        data_file = join(args['<reduced_data_folder>'], "reduced.csv")
    working_dir_path = create_working_directory(args['--out'], "retrieval_results/", model_type)
    # Load the reduced data
    data = DataContainer(data_file)
    print("Cleaning up the data first...")
    common.preprocess_data(data)
    X, _, _ = data.get_data()

    datasetIDs = data.get_dataset_IDs()
    labels = data.get_labels()

    summary_csv_file = open(join(working_dir_path, "retrieval_summary.csv"), 'w')
    # Write out the file headers
    summary_csv_file.write('dataset,celltype,#cell,mean average precision\n')

    sorted_unique_datasetIDS = np.unique(datasetIDs)
    for dataset in sorted_unique_datasetIDS:
        # We will only compare samples from different datasets, so separate them
        current_ds_samples_indicies = np.where(datasetIDs == dataset)[0]
        current_ds_samples = X[current_ds_samples_indicies]
        other_ds_samples_indicies = np.where(datasetIDs != dataset)[0]
        other_ds_samples = X[other_ds_samples_indicies]
        distance_matrix = distance.cdist(current_ds_samples, other_ds_samples, metric=args['--dist_metric'])

        average_precisions_for_label = defaultdict(list)

        for index, distances in enumerate(distance_matrix):
            current_sample_idx = current_ds_samples_indicies[index]
            current_sample_label = labels[current_sample_idx]
            if current_sample_label not in common.CLEAN_LABEL_SUBSET:
                continue
            sorted_distances_indicies = np.argsort(distances)

            # Count the total number of same label samples in other datasets
            total_same_label = 0
            for i in range(len(distances)):
                label = labels[other_ds_samples_indicies[i]]
                if label == current_sample_label:
                    total_same_label += 1

            retrieved_labels = []
            for retrieved_idx in sorted_distances_indicies[:total_same_label]:
                retrieved_labels.append(labels[other_ds_samples_indicies[retrieved_idx]])
            avg_precision = average_precision(current_sample_label, retrieved_labels)
            average_precisions_for_label[current_sample_label].append(avg_precision)

        for label, average_precisions in average_precisions_for_label.items():
            num_samples = len(average_precisions)
            summary_csv_file.write(str(dataset) + ',' + label + ',' + str(num_samples) + ',' + str(np.mean(average_precisions)) + '\n')

    summary_csv_file.close()
