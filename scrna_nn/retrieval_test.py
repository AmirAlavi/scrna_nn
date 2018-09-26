import pickle
from collections import defaultdict
from os.path import join

import numpy as np
from scipy.spatial import distance

from .data_manipulation.data_container import DataContainer
from .util import create_working_directory, distances


# def average_accuracy(query_label, retrieved_labels, dist_mat_by_strings, max_dist):
#     avg_acc = 0
#     for r in retrieved_labels:
#         avg_acc += max(0, 1 - (dist_mat_by_strings[query_label][r] / max_dist))
#     return avg_acc/len(retrieved_labels)

def average_flex_precision(query_label, retrieved_labels, similarity_fcn, is_asymm):
    """The difference between this and 'average_flex_precision' is that while that function
    only calculated a score at recall positions that were perfect matches, this one calculates
    at every position. (the other function will return 0 if there were no perfect matches).
    """
    scores = []
    relevance_sum = 0
    for idx, retrieved in enumerate(retrieved_labels):
        if retrieved == query_label:
            relevance_sum += 1
            scores.append(relevance_sum / float(idx+1))
        else:
            try:
                relevance = similarity_fcn(query_label, retrieved)
                if is_asymm:
                    relevance = min(relevance, similarity_fcn(retrieved, query_label))
                relevance_sum += relevance
                scores.append(relevance_sum / float(idx+1))
            except KeyError as err:
                # no similarity information available (not in dict), skip
                pass
        
    if len(scores) > 0:
        return np.mean(scores)
    else:
        return 0.0

# def average_flex_precision(query_label, retrieved_labels, dist_mat_by_strings, max_dist):
#     scores = []
#     relevance_sum = 0
#     for idx, retrieved in enumerate(retrieved_labels):
#         relevance = max(0, 1 - (dist_mat_by_strings[query_label][retrieved] / max_dist))
#         relevance_sum += relevance
#         if retrieved == query_label:
#             scores.append(relevance_sum / float(idx+1))
#     if len(scores) > 0:
#         return np.mean(scores)
#     else:
#         return 0.0
        
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

def retrieval_test_in_memory(db, db_labels, query, query_labels):
    similarity_fcn = distances.TextMinedPairSimilarity(distance_mat_file='dump_A_1.p',
                                                       transform='linear',
                                                       transform_param=1)
    # Find out the number of results to return.
    db_uniq, db_counts = np.unique(db_labels, return_counts=True)
    query_uniq, query_counts = np.unique(query_labels, return_counts=True)
    query_label_count_d = {label: count for (label, count) in zip(query_uniq, query_counts)}
    db_label_count_d = {label: count for (label, count) in zip(db_uniq, db_counts)}

    counts_in_db_for_query_labels = []
    for query_label, query_count in zip(query_uniq, query_counts):
        count_in_db = db_label_count_d[query_label]
        counts_in_db_for_query_labels.append(count_in_db)
    min_db_label_count = min(counts_in_db_for_query_labels)
    num_results = min(100, min_db_label_count)

    average_precisions_for_label = defaultdict(list)
    average_flex_precisions_for_label = defaultdict(list)

    distance_matrix = distance.cdist(query, db, metric='euclidean')
    for index, distances_to_query in enumerate(distance_matrix): # Loop is over the set of query cells
        query_label = query_labels[index]
        sorted_distances_indices = np.argsort(distances_to_query)
        retrieved_labels_sorted_by_distance = db_labels[sorted_distances_indices]
        retrieved_labels = retrieved_labels_sorted_by_distance[:num_results]
        avg_flex_precision = average_flex_precision(query_label, retrieved_labels, similarity_fcn, True)
        avg_precision = average_precision(query_label, retrieved_labels)
        average_precisions_for_label[query_label].append(avg_precision)
        average_flex_precisions_for_label[query_label].append(avg_flex_precision)

    retrieval_results_d = {"cell_types":{}}
    maps = [] # mean average precisions
    mafps = [] # mean average flex precisions
    weights = []
    for label in average_precisions_for_label.keys():
        average_precisions = average_precisions_for_label[label]
        average_flex_precisions = average_flex_precisions_for_label[label]
        cur_map = np.mean(average_precisions)
        maps.append(cur_map)
        cur_mafp = np.mean(average_flex_precisions)
        mafps.append(cur_mafp)
        cur_weight = query_label_count_d[label]
        weights.append(cur_weight)
        retrieval_results_d["cell_types"][label] = {"#_in_query": cur_weight, "#_in_DB": db_label_count_d[label], "Mean_Average_Precision": cur_map, "Mean_Average_Flex_Precision": cur_mafp}

    retrieval_results_d["average_map"] = np.mean(maps)
    retrieval_results_d["weighted_average_map"] = np.average(maps, weights=weights)

    retrieval_results_d["average_mafp"] = np.mean(mafps)
    retrieval_results_d["weighted_average_mafp"] = np.average(mafps, weights=weights)

    return retrieval_results_d["average_map"], retrieval_results_d["weighted_average_map"], retrieval_results_d["average_mafp"], retrieval_results_d["weighted_average_mafp"]
    
def retrieval_test(args):
    if args.similarity_type == 'ontology':
        print("ontology-based similarities")
        similarity_fcn = distances.OntologyBasedPairSimilarity(max_ontology_distance=args.max_ont_path_len,
                                                               distance_mat_file=args.sim_mat_file,
                                                               transform=args.sim_trnsfm_fcn,
                                                               transform_param=args.sim_trnsfm_param)
    elif args.similarity_type == 'text-mined':
        print("text-mined similarities")
        similarity_fcn = distances.TextMinedPairSimilarity(distance_mat_file=args.sim_mat_file,
                                                           transform=args.sim_trnsfm_fcn,
                                                           transform_param=args.sim_trnsfm_param)
    else:
        raise ScrnaException("Not a valid similarity type!")


    
    working_dir_path = create_working_directory(args.out, "retrieval_results/")
    # Load the reduced data
    query_data = DataContainer(args.query_data_file)
    database_data = DataContainer(args.database_data_file)
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
        data_summary_f.write("Query data file: " + args.query_data_file)
        data_summary_f.write("Database data file: " + args.database_data_file)
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
    average_flex_precisions_for_label = defaultdict(list)
    all_average_precisions = []
    all_average_flex_precisions = []
    # average_flex_precisions_for_label2 = defaultdict(list)
    # average_accuracies_for_label = defaultdict(list)
    # average_top_fourth_accuracies_for_label = defaultdict(list)
    distance_matrix = distance.cdist(queries, db, metric=args.dist_metric)
    for index, distances_to_query in enumerate(distance_matrix): # Loop is over the set of query cells
        query_label = queries_labels[index]
        sorted_distances_indices = np.argsort(distances_to_query)
        retrieved_labels_sorted_by_distance = db_labels[sorted_distances_indices]
        retrieved_labels = retrieved_labels_sorted_by_distance[:num_results]
        # avg_accuracy = average_accuracy(query_label, retrieved_labels, dist_mat_by_strings, int(args['--max_ont_dist']))
        # top_fourth_idx = int(num_results/4)
        # avg_accuracy_of_top_fourth = average_accuracy(query_label, retrieved_labels[:top_fourth_idx], dist_mat_by_strings, int(args['--max_ont_dist']))
        avg_flex_precision = average_flex_precision(query_label, retrieved_labels, similarity_fcn, args.asymm_dist)
        #  avg_flex_precision2 = average_flex_precision2(query_label, retrieved_labels, dist_mat_by_strings, int(args['--max_ont_dist']))
        avg_precision = average_precision(query_label, retrieved_labels)
        if avg_flex_precision <= 0.2:
            print("\tLOW SCORE")
            print("\tQuery label: ", query_label)
            print("\tRetrieved: ")
            for l in retrieved_labels:
                print("\t\t" + l)
        average_precisions_for_label[query_label].append(avg_precision)
        average_flex_precisions_for_label[query_label].append(avg_flex_precision)
        all_average_precisions.append("{},{}".format(query_label, avg_precision))
        all_average_flex_precisions.append("{},{}".format(query_label, avg_flex_precision))
        # average_flex_precisions_for_label2[query_label].append(avg_flex_precision2)
        # average_accuracies_for_label[query_label].append(avg_accuracy)
        # average_top_fourth_accuracies_for_label[query_label].append(avg_accuracy_of_top_fourth)

    retrieval_results_d = {"cell_types":{}}
    maps = [] # mean average precisions
    mafps = [] # mean average flex precisions
    # mafp2s = [] # mean average flex precisions (2)
    # macs = [] # mean average accuracies
    # macqs = [] # mean average accuracies of top quarter
    weights = []
    for label in average_precisions_for_label.keys():
        average_precisions = average_precisions_for_label[label]
        average_flex_precisions = average_flex_precisions_for_label[label]
        # average_flex_precisions2 = average_flex_precisions_for_label2[label]
        # average_accuracies = average_accuracies_for_label[label]
        # average_fourth_accuracies = average_top_fourth_accuracies_for_label[label]

        cur_map = np.mean(average_precisions)
        maps.append(cur_map)
        cur_mafp = np.mean(average_flex_precisions)
        mafps.append(cur_mafp)
        # cur_mafp2 = np.mean(average_flex_precisions2)
        # mafp2s.append(cur_mafp2)
        # cur_mac = np.mean(average_accuracies)
        # macs.append(cur_mac)
        # cur_macq = np.mean(average_fourth_accuracies)
        # macqs.append(cur_macq)
        cur_weight = query_label_count_d[label]
        weights.append(cur_weight)
        # retrieval_results_d["cell_types"][label] = {"#_in_query": cur_weight, "#_in_DB": db_label_count_d[label], "Mean_Average_Precision": cur_map, "Mean_Average_Flex_Precision": cur_mafp, "Mean_Average_Flex_Precision2": cur_mafp2, "Mean_Average_Accuracy": cur_mac, "Mean_Average_Accuracy_of_top_quarter": cur_macq}
        retrieval_results_d["cell_types"][label] = {"#_in_query": cur_weight, "#_in_DB": db_label_count_d[label], "Mean_Average_Precision": cur_map, "Mean_Average_Flex_Precision": cur_mafp}

    retrieval_results_d["average_map"] = np.mean(maps)
    retrieval_results_d["weighted_average_map"] = np.average(maps, weights=weights)

    retrieval_results_d["average_mafp"] = np.mean(mafps)
    retrieval_results_d["weighted_average_mafp"] = np.average(mafps, weights=weights)

    # retrieval_results_d["average_mafp2"] = np.mean(mafp2s)
    # retrieval_results_d["weighted_average_mafp2"] = np.average(mafp2s, weights=weights)

    # retrieval_results_d["average_mac"] = np.mean(macs)
    # retrieval_results_d["weighted_average_mac"] = np.average(macs, weights=weights)

    # retrieval_results_d["average_macq"] = np.mean(macqs)
    # retrieval_results_d["weighted_average_macq"] = np.average(macqs, weights=weights)
    with open(join(working_dir_path, "retrieval_results_d.pickle"), 'wb') as f:
        pickle.dump(retrieval_results_d, f)

    with open(join(working_dir_path, "all_avg_precision_scores.txt"), 'w') as f:
        for score in all_average_precisions:
            f.write(score + '\n')
        
    with open(join(working_dir_path, "all_avg_flex_precision_scores.txt"), 'w') as f:
        for score in all_average_flex_precisions:
            f.write(score + '\n')
        
        
    # summary_csv_file = open(join(working_dir_path, "retrieval_summary.csv"), 'w')
    # with open(join(working_dir_path, "retrieval_summary.csv"), 'w') as summary_csv_file:
    #     # Write out the file headers
    #     summary_csv_file.write('celltype\t#query cells\tmean average precision\n')
    #     for label, average_precisions in average_precisions_for_label.items():
    #         num_samples = len(average_precisions)
    #         summary_csv_file.write(label + '\t' + str(num_samples) + '\t' + str(np.mean(average_precisions)) + '\n')
