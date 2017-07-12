from collections import defaultdict

import numpy as np

def get_groupings_for_genes(ppi_tf_groups_filepath, dataset_gene_names):
    '''Get the PPI/TF membership of each of the genes in the dataset.

    Args:
        ppi_tf_groups_filepath: Path to file containing the TF groups and PPI
            groups, each on separate lines.
        dataset_gene_names: List of string gene names that are present in the
            dataset (feature space of interest).

    Returns:
        ppi_tf_groups_as_indicies: A dictionary <string, list> of ppi/tf group
            names --> list of indicies into "dataset_gene_names".
        sorted_group_names: Sorted list of ppi/tf group names. Corresponds to
            each row of "binary_group_membership_mat".
        binary_group_membership_mat: Matrix with each row being a PPI or TF
            group, and each column being an index into "dataset_gene_names",
            where a 1 indicates that that particular column (gene) is in the
            ppi/tf group represented by that row. 0 otherwise. It is a sorted
            matrix representation of "ppi_tf_groups_as_indicies".
    '''
    dataset_gene_names = dataset_gene_names.tolist()
    print("num gene names: ", len(dataset_gene_names))
    lines = open(ppi_tf_groups_filepath).readlines()
    ppi_tf_groups_as_indices = []
    group_names = []
    largest_index = -1
    for line in lines:
        # Get tab separated tokens in the line
        tokens = line.replace('\n', '').replace('\r', '').split('\t')
        # The first token is the name of that group (either 'TF tfname' or
        # 'ppi_groupnumber')
        group_names.append(tokens[0])
        indices_list = []
        # The rest of the tokens are the names of the genes in that group
        for gene in tokens[1:]:
            if gene in dataset_gene_names:# Decoupling the set of genes in dataset from set of genes in PPITF knowledge
                #ppi_tf_groups_as_indices[group_name].append(dataset_gene_names.index(gene))
                idx = dataset_gene_names.index(gene)
                if idx > largest_index:
                    largest_index = idx
                indices_list.append(idx)
        ppi_tf_groups_as_indices.append(indices_list)
    #largest_index = max(map(max, ppi_tf_groups_as_indices.values()))
    #sorted_group_names = sorted(ppi_tf_groups_as_indices.keys())
    binary_group_membership_mat = np.zeros((largest_index+1, len(group_names)), dtype='float32')
    for group_idx in range(len(group_names)):
        for gene_idx in ppi_tf_groups_as_indices[group_idx]:
            binary_group_membership_mat[gene_idx, group_idx] = 1
    # for group_idx, group_name in enumerate(sorted_group_names):
    #     for gene_idx in ppi_tf_groups_as_indices[group_name]:
    #         binary_group_membership_mat[group_idx, gene_idx] = 1
    return ppi_tf_groups_as_indices, group_names, binary_group_membership_mat

'''
def get_groupings_for_genes_new(dataset_gene_names, tf_master_list, ppi_master_list):
    ppi_tf_groups_as_indices = defaultdict(list)
    # build TF groups
    #tf_groups = defaultdict(list)
    with open(tf_master_list, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]: # file has a header
        stripped = line.rstrip().lower().split('\t')
        tf = 'tf_' + stripped[0]
        gene = stripped[1].split(';')[0]
        if gene in dataset_gene_names:
            #tf_groups[tf].append(gene)
            ppi_tf_groups_as_indices[tf].append(dataset_gene_names.index(gene))
    # build PPI groups
    #ppi_groups = defaultdict(list)
    with open(ppi_master_list, 'r') as f:
        lines = f.readlines()
    count = 0
    for line in lines:
        stripped = line.rstrip().lower().split('\t')
        ppi = 'ppi_' + str(count)
        for gene in stripped:
            if gene in dataset_gene_names:
                #ppi_groups[tf].append(gene)
                ppi_tf_groups_as_indices[tf].append(dataset_gene_names.index(gene))
        count += 1
    largest_index = max(map(max, ppi_tf_groups_as_indices.values()))
    sorted_group_names = sorted(ppi_tf_groups_as_indices.keys())
    binary_group_membership_mat = np.zeros((len(sorted_group_names), largest_index+1), dtype='float32')
    for group_idx, group_name in enumerate(sorted_group_names):
        for gene_idx in ppi_tf_groups_as_indices[group_name]:
            binary_group_membership_mat[group_idx, gene_idx] = 1
    return ppi_tf_groups_as_indices, sorted_group_names, binary_group_membership_mat
''' 
