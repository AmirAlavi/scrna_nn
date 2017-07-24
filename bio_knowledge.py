from collections import defaultdict

import numpy as np

def get_adj_mat_from_groupings(groups_filepath, dataset_gene_names):
    '''Get the group membership of each of the genes in the dataset.

    Args:
        groups_filepath: Path to file containing the gene groupings
            each on separate lines.
        dataset_gene_names: List of string gene names that are present in the
            dataset (feature space of interest).

    Returns:
        groups_as_indices: A list of lists of indicies into "dataset_gene_names".
        group_names: List of the group names. Corresponds to
            each row of "binary_group_membership_mat".
        binary_group_membership_mat: Matrix with each column being a grouping,
            and each row being an index into "dataset_gene_names",
            where a 1 indicates that that particular row (gene) is in the
            group represented by that row. 0 otherwise. It is a
            matrix representation of "groups_as_indices".
    '''
    dataset_gene_names = dataset_gene_names.tolist()
    num_genes = len(dataset_gene_names)
    print("num gene names: ", num_genes)
    lines = open(groups_filepath).readlines()
    groups_as_indices = []
    group_names = []
    for line in lines:
        # Get tab separated tokens in the line
        tokens = line.replace('\n', '').replace('\r', '').split('\t')
        # The first token is the name of that group (e.g. 'TF tfname' or
        # 'ppi_groupnumber' or 'GO:go_id')
        group_names.append(tokens[0])
        indices_list = []
        # The rest of the tokens are the names of the genes in that group
        for gene in tokens[1:]:
            if gene in dataset_gene_names:# Decoupling the set of genes in dataset from set of genes in the groupings file
                #groups_as_indices[group_name].append(dataset_gene_names.index(gene))
                idx = dataset_gene_names.index(gene)
                indices_list.append(idx)
        groups_as_indices.append(indices_list)
    binary_group_membership_mat = np.zeros((num_genes, len(group_names)), dtype='float32')
    for group_idx in range(len(group_names)):
        for gene_idx in groups_as_indices[group_idx]:
            binary_group_membership_mat[gene_idx, group_idx] = 1
    return groups_as_indices, group_names, binary_group_membership_mat
