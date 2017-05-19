from collections import defaultdict

def get_groupings_for genes(ppi_tf_groups_filepath, dataset_gene_names):
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
