from collections import defaultdict

def get_groupings_for genes(ppi_tf_groups_filepath, dataset_gene_names):
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
    lines = open(ppi_tf_groups_filepath).readlines()
    ppi_tf_groups_as_indicies = defaultdict(list)
    for line in lines:
        # Get tab separated tokens in the line
        tokens = line.replace('\n', '').replace('\r', '').split('\t')
        # The first token is the name of that group (either 'TF tfname' or
        # 'ppi_groupnumber')
        group_name = tokens[0]
        # The rest of the tokens are the names of the genes in that group
        for gene in tokens[1:]:
            ppi_tf_groups_as_indicies[group_name].append(dataset_gene_names.index(gene))
        largest_index = max(map(max, ppi_tf_groups_as_indicies.values()))
        sorted_group_names = sorted(ppi_tf_groups_as_indicies.keys())
        binary_group_membership_mat = np.zeros((len(sorted_group_names), largest_index+1), dtype='float32')
        for group_idx, group_name in enumerate(sorted_group_names):
            for gene_idx in ppi_tf_groups_as_indicies[group_name]:
                binary_group_membership_mat[group_idx, gene_idx] = 1
        return ppi_tf_groups_as_indicies, sorted_group_names, binary_group_membership_mat
