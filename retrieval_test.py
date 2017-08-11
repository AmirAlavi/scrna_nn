import pickle
from collections import defaultdict
from os.path import join

import numpy as np
from scipy.spatial import distance

from data_container import DataContainer
from util import create_working_directory


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
    queries = query_data.get_expression_mat()
    db = database_data.get_expression_mat()
    queries_labels = query_data.get_labels()
    db_labels = database_data.get_labels()

    # Find out the number of results to return.
    db_uniq, db_counts = np.unique(db_labels, return_counts=True)
    query_uniq, query_counts = np.unique(queries_labels, return_counts=True)
    query_label_count_d = {label: count for (label, count) in zip(query_uniq, query_counts)}
    db_label_count_d = {label: count for (label, count) in zip(db_uniq, db_counts)}
    min_db_label_count = np.amin(db_counts)
    num_results = min(100, min_db_label_count)

    with open(join(working_dir_path, "data_summary.txt"), 'w') as data_summary_f:
        data_summary_f.write("Query data file: " + args['<query_data_file>'])
        data_summary_f.write("Database data file: " + args['<database_data_file>'])
        data_summary_f.write("num query points: " + str(len(queries_labels)) + '\n')
        data_summary_f.write("num database points: " + str(len(db_labels)) + '\n')
        data_summary_f.write("num query types: " + str(len(query_counts)) + '\n')
        data_summary_f.write("num database types: " + str(len(db_counts)) + '\n')
        data_summary_f.write("\nmin number of cells of any single type in DB: " + str(min_db_label_count) + '\n')
        data_summary_f.write("\nLabel\t#Query\t#DB\n")
        for query_label, query_count in zip(query_uniq, query_counts):
            data_summary_f.write(query_label + '\t' + str(query_count) + '\t' + str(db_label_count_d[query_label]) + '\n')


    
    average_precisions_for_label = defaultdict(list)
    distance_matrix = distance.cdist(queries, db, metric=args['--dist_metric'])
    for index, distances_to_query in enumerate(distance_matrix): # Loop is over the set of query cells
        query_label = queries_labels[index]
        sorted_distances_indices = np.argsort(distances_to_query)
        retrieved_labels_sorted_by_distance = db_labels[sorted_distances_indices]
        retrieved_labels = retrieved_labels_sorted_by_distance[:num_results]
        avg_precision = average_precision(query_label, retrieved_labels)
        average_precisions_for_label[query_label].append(avg_precision)

    retrieval_results_d = {"cell_types":{}}
    maps = []
    weights = []
    for label, average_precisions in average_precisions_for_label.items():
        cur_map = np.mean(average_precisions)
        maps.append(cur_map)
        cur_weight = query_label_count_d[label]
        weights.append(cur_weight)
        retrieval_results_d["cell_types"][label] = {"#_in_query": cur_weight, "#_in_DB": db_label_count_d[label], "Mean_Average_Precision": cur_map}
    retrieval_results_d["average"] = np.mean(maps)
    retrieval_results_d["weighted_average"] = np.average(maps, weights=weights)
    with open(join(working_dir_path, "retrieval_results_d.pickle"), 'wb') as f:
        pickle.dump(retrieval_results_d, f)
        
    # summary_csv_file = open(join(working_dir_path, "retrieval_summary.csv"), 'w')
    # with open(join(working_dir_path, "retrieval_summary.csv"), 'w') as summary_csv_file:
    #     # Write out the file headers
    #     summary_csv_file.write('celltype\t#query cells\tmean average precision\n')
    #     for label, average_precisions in average_precisions_for_label.items():
    #         num_samples = len(average_precisions)
    #         summary_csv_file.write(label + '\t' + str(num_samples) + '\t' + str(np.mean(average_precisions)) + '\n')
