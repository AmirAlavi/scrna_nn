import pickle
from collections import defaultdict
from os.path import join

import numpy as np
from scipy.spatial import distance

from data_container import DataContainer
from util import create_working_directory

def average_accuracy(query_label, retrieved_labels, dist_mat_by_strings, max_dist):
    avg_acc = 0
    for r in retrieved_labels:
        avg_acc += max(0, 1 - (dist_mat_by_strings[query_label][r] / max_dist))
    return avg_acc/len(retrieved_labels)

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
    with open(args['--dist_mat_file'], 'rb') as f:
        dist_mat_by_strings = pickle.load(f)
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

    with open(join(working_dir_path, "data_summary.txt"), 'w') as data_summary_f:
        data_summary_f.write("Query data file: " + args['<query_data_file>'])
        data_summary_f.write("Database data file: " + args['<database_data_file>'])
        data_summary_f.write("num query points: " + str(len(queries_labels)) + '\n')
        data_summary_f.write("num database points: " + str(len(db_labels)) + '\n')
        data_summary_f.write("num query types: " + str(len(query_counts)) + '\n')
        data_summary_f.write("num database types: " + str(len(db_counts)) + '\n')
        data_summary_f.write("\nLabel\t#Query\t#DB\n")
        counts_in_db_for_query_labels = []
        for query_label, query_count in zip(query_uniq, query_counts):
            count_in_db = db_label_count_d[query_label]
            counts_in_db_for_query_labels.append(count_in_db)
            data_summary_f.write(query_label + '\t' + str(query_count) + '\t' + str(count_in_db) + '\n')
        min_db_label_count = min(counts_in_db_for_query_labels)
        data_summary_f.write("\nmin number of cells of any single type in DB: " + str(min_db_label_count) + '\n')
        num_results = min(100, min_db_label_count)

    average_precisions_for_label = defaultdict(list)
    average_accuracies_for_label = defaultdict(list)
    average_top_fourth_accuracies_for_label = defaultdict(list)
    distance_matrix = distance.cdist(queries, db, metric=args['--dist_metric'])
    for index, distances_to_query in enumerate(distance_matrix): # Loop is over the set of query cells
        query_label = queries_labels[index]
        sorted_distances_indices = np.argsort(distances_to_query)
        retrieved_labels_sorted_by_distance = db_labels[sorted_distances_indices]
        retrieved_labels = retrieved_labels_sorted_by_distance[:num_results]
        avg_accuracy = average_accuracy(query_label, retrieved_labels, dist_mat_by_strings, int(args['--max_dist']))

        avg_accuracy_of_top_fourth = average_accuracy(query_label, retrieved_labels[:num_results/4], dist_mat_by_strings, int(args['--max_dist']))
        avg_precision = average_precision(query_label, retrieved_labels)
        if avg_precision <= 0.1:
            print("\tLOW SCORE")
            print("\tQuery label: ", query_label)
            print("\tRetrieved: ")
            for l in retrieved_labels:
                print("\t\t" + l)
        average_precisions_for_label[query_label].append(avg_precision)
        average_accuracies_for_label[query_label].append(avg_accuracy)
        average_top_fourth_accuracies_for_label[query_label].append(avg_accuracy_of_top_fourth)

    retrieval_results_d = {"cell_types":{}}
    maps = [] # mean average precisions
    macs = [] # mean average accuracies
    macqs = [] # mean average accuracies of top quarter
    weights = []
    for label in average_precisions_for_label.keys():
        average_precisions = average_precisions_for_label[label]
        average_accuracies = average_accuracies_for_label[label]
        average_fourth_accuracies = average_top_fourth_accuracies_for_label[label]

        cur_map = np.mean(average_precisions)
        maps.append(cur_map)
        cur_mac = np.mean(average_accuracies)
        macs.append(cur_mac)
        cur_macq = np.mean(average_fourth_accuracies)
        macqs.append(cur_macq)
        cur_weight = query_label_count_d[label]
        weights.append(cur_weight)
        retrieval_results_d["cell_types"][label] = {"#_in_query": cur_weight, "#_in_DB": db_label_count_d[label], "Mean_Average_Precision": cur_map, "Mean_Average_Accuracy": cur_mac, "Mean_Average_Accuracy_of_top_quarter": cur_macq}

    retrieval_results_d["average_map"] = np.mean(maps)
    retrieval_results_d["weighted_average_map"] = np.average(maps, weights=weights)

    retrieval_results_d["average_mac"] = np.mean(macs)
    retrieval_results_d["weighted_average_mac"] = np.average(macs, weights=weights)

    retrieval_results_d["average_macq"] = np.mean(macqs)
    retrieval_results_d["weighted_average_macq"] = np.average(macqs, weights=weights)
    with open(join(working_dir_path, "retrieval_results_d.pickle"), 'wb') as f:
        pickle.dump(retrieval_results_d, f)
        
    # summary_csv_file = open(join(working_dir_path, "retrieval_summary.csv"), 'w')
    # with open(join(working_dir_path, "retrieval_summary.csv"), 'w') as summary_csv_file:
    #     # Write out the file headers
    #     summary_csv_file.write('celltype\t#query cells\tmean average precision\n')
    #     for label, average_precisions in average_precisions_for_label.items():
    #         num_samples = len(average_precisions)
    #         summary_csv_file.write(label + '\t' + str(num_samples) + '\t' + str(np.mean(average_precisions)) + '\n')
