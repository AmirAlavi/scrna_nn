import random
from os import makedirs
from os.path import join, exists
from itertools import combinations

import numpy as np

from . import distances
from . import util
from .util import ScrnaException

CACHE_ROOT = "_cache"
SIAM_CACHE = "siam_data"

def get_bucket_boundaries(sorted_tuple_list, n_buckets):
    step_size = (sorted_tuple_list[-1][0] - sorted_tuple_list[0][0]) / n_buckets
    boundaries = []
    left_idx = 0
    right_idx = 0
    for i in range(n_buckets-1):
        while right_idx < len(sorted_tuple_list) and sorted_tuple_list[right_idx][0] < sorted_tuple_list[0][0] + ((i+1)*step_size):
            right_idx += 1
        boundaries.append((left_idx, right_idx))
        left_idx = right_idx
    boundaries.append((left_idx, len(sorted_tuple_list)))
    return boundaries

def uniformly_select_diff_pairs(n_buckets, candidate_diff_samples, X, y, true_ids, anchor_samples, same_count, diff_pairs_multiplier):
    pairs = []
    labels = []
    sorted_candidates = sorted(candidate_diff_samples, key=lambda tup: tup[0])
    bucket_boundaries = get_bucket_boundaries(sorted_candidates, n_buckets)
    for boundaries in bucket_boundaries:
        subset = sorted_candidates[boundaries[0]:boundaries[1]]
        random.shuffle(subset)
        num_to_take = min(len(subset), int(diff_pairs_multiplier*same_count/n_buckets))
        for i in range(num_to_take):
            # select a random anchor sample
            anchor_idx = random.choice(anchor_samples)
            diff_idx = subset[i][2]
            similarity = subset[i][1] # use the transformed similarity
            while(true_ids[anchor_idx] == true_ids[diff_idx]):
                # for the current different sample, be sure they aren't the same underlying sample
                anchor_idx = random.choice(anchor_samples)
            anchor_vec = X[anchor_idx]
            diff_vec = X[diff_idx]
            pairs += [[ anchor_vec, diff_vec ]]
            labels += [similarity]
    return pairs, labels

def unconstrained_select_diff_pairs(candidate_diff_samples, X, y, true_ids, anchor_samples, same_count, diff_pairs_multiplier):
    pairs = []
    labels = []
    random.shuffle(candidate_diff_samples)
    num_to_take = min(len(candidate_diff_samples), diff_pairs_multiplier*same_count)
    for i in range(num_to_take):
        # select a random anchor sample
        anchor_idx = random.choice(anchor_samples)
        diff_idx = candidate_diff_samples[i][2]
        similarity = candidate_diff_samples[i][1] # use the transformed similarity
        while(true_ids[anchor_idx] == true_ids[diff_idx]):
            # for the current different sample, be sure they aren't the same underlying sample
            anchor_idx = random.choice(anchor_samples)
        anchor_vec = X[anchor_idx]
        diff_vec = X[diff_idx]
        pairs += [[ anchor_vec, diff_vec ]]
        labels += [similarity]
    return pairs, labels

def select_diff_pairs(X, y, true_ids, label_strings_lookup, anchor_label, anchor_samples, indices_lists, similarity_fcn, same_count, args):
    similarity_to_anchor = [] # elements will be tuples of (raw_similarity, transformed_similarity, diff_sample_id) where similarity is the similarity b/w this diff sample and the anchor sample
    for diff_label, diff_samples in indices_lists.items():
        if diff_label == anchor_label:
            continue # picked a cell type that is same as anchor type, we want one that is different
        anchor_string = label_strings_lookup[anchor_label]
        diff_string = label_strings_lookup[diff_label]
        raw_similarity = similarity_fcn(anchor_string, diff_string, transform=False)
        transformed_similarity = similarity_fcn(anchor_string, diff_string, transform=True)
        for s in diff_samples:
            similarity_to_anchor.append((raw_similarity, transformed_similarity, s))
    if int(args['--unif_diff']) > 0:
        print("uniform!")
        pairs, labels = uniformly_select_diff_pairs(int(args['--unif_diff']), similarity_to_anchor, X, y, true_ids, anchor_samples, same_count, int(args['--diff_multiplier']))
    else:
        print("unconstrained!")
        pairs, labels = unconstrained_select_diff_pairs(similarity_to_anchor, X, y, true_ids, anchor_samples, same_count, int(args['--diff_multiplier']))
    return pairs, labels

#def create_flexible_data_pairs(X, y, true_ids, indices_lists, same_lim, label_strings_lookup, args):
def create_flexible_data_pairs(X, y, true_ids, label_strings_lookup, args):
    # normalization = ""
    # if args['--sn']:
    #     normalization = "sn"
    # elif args['--gn']:
    #     normalization = "gn"
    # config_string = '_'.join([args['--data'], normalization, args['--dynMarginLoss'], args['--dist_mat_file'], args['--trnsfm_fcn'], args['--trnsfm_fcn_param'], args['--unif_diff'], args['--same_lim'], args['--diff_multiplier']])
    # cache_path = join(join(CACHE_ROOT, SIAM_CACHE), config_string)
    # if exists(cache_path):
    #     print("Loading siamese data from cache...")
    #     pairs = np.load(join(cache_path, "siam_X.npy"))
    #     labels = np.load(join(cache_path, "siam_y.npy"))
    #     return pairs, labels
    same_lim = int(args['--same_lim'])
    if args['--dynMarginLoss'] == 'ontology':
        print("ontology-based similarities")
        similarity_fcn = distances.OntologyBasedPairSimilarity(max_ontology_distance=int(args['--max_ont_dist']),
                                                               distance_mat_file=args['--dist_mat_file'],
                                                               transform=args['--trnsfm_fcn'],
                                                               transform_param=int(args['--trnsfm_fcn_param']))
    elif args['--dynMarginLoss'] == 'text-mined':
        print("text-mined similarities")
        similarity_fcn = distances.TextMinedPairSimilarity(distance_mat_file=args['--dist_mat_file'],
                                                           transform=args['--trnsfm_fcn'],
                                                           transform_param=int(args['--trnsfm_fcn_param']))
    else:
        raise ScrnaException("Not a valid dynMarginLoss type!")

    print("Generating 'Flexible' pairs for siamese")
    pairs = []
    labels = []
    indices_lists = util.build_indices_master_list(X, y)

    for anchor_label, anchor_samples in indices_lists.items():
        same_count = 0
        combs = combinations(anchor_samples, 2)
        # TODO: should I shuffle the combs?
        for comb in combs:
            pairs += [[ X[comb[0]], X[comb[1]] ]]
            labels += [1]
            same_count += 1
            if same_count == same_lim:
                break
        # create the different pairs
        diff_pairs, diff_labels = select_diff_pairs(X, y, true_ids, label_strings_lookup, anchor_label, anchor_samples, indices_lists, similarity_fcn, same_count, args)
        pairs += diff_pairs
        labels += diff_labels

    print("Generated ", len(pairs), " pairs")
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print("Distribution of pairs labels: ")
    print(unique_labels)
    print(label_counts)
    pairs_np = np.array(pairs)
    labels_np = np.array(labels)
    # makedirs(cache_path)
    # np.save(join(cache_path, "siam_X"), pairs_np)
    # np.save(join(cache_path, "siam_y"), labels_np)
    return pairs_np, labels_np
