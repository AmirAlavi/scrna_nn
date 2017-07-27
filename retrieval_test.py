from os.path import join
import json
from collections import defaultdict

import numpy as np
from scipy.spatial import distance

from util import create_working_directory
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
    working_dir_path = create_working_directory(args['--out'], "retrieval_results/")
    # Load the reduced data
    query_data = DataContainer(args['<query_data_file>'])
    database_data = DataContainer(args['<database_data_file>'])
    queries, _, _ = query_data.get_data()
    db, _, _ = database_data.get_data()
    queries_labels = query_data.get_labels()
    db_labels = database_data.get_labels()
    print("num query points = ", len(queries_labels))
    print("num database points = ", len(db_labels))

    average_precisions_for_label = defaultdict(list)
    distance_matrix = distance.cdist(queries, db, metric=args['--dist_metric'])
    for index, distances_to_query in enumerate(distance_matrix):
        query_label = queries_labels[index]
        sorted_distances_indices = np.argsort(distances_to_query)
        retrieved_labels_sorted_by_distance = db_labels[sorted_distances_indices]
        retrieved_labels = retrieved_labels_sorted_by_distance[:100]
        avg_precision = average_precision(query_label, retrieved_labels)
        average_precisions_for_label[query_label].append(avg_precision)

    summary_csv_file = open(join(working_dir_path, "retrieval_summary.csv"), 'w')
    with open(join(working_dir_path, "retrieval_summary.csv"), 'w') as summary_csv_file:
        # Write out the file headers
        summary_csv_file.write('celltype\t#query cells\tmean average precision\n')
        for label, average_precisions in average_precisions_for_label.items():
            num_samples = len(average_precisions)
            summary_csv_file.write(label + '\t' + str(num_samples) + '\t' + str(np.mean(average_precisions)) + '\n')
