"""Data Analysis & Preparation Script (for scRNA Pipeline)

Usage:
    data_prep.py <expression_h5> <ontology_mappings_json> [--uniq_lvl=<uniqueness>]
    data_prep.py (-h | --help)
    data_prep.py --version

Options:
    --uniq_lvl=<uniqueness>    How unique the ontology mappings for each cell should be.
                               <uniqueness> is a float between [0.0, 1.0] and is a minimum Jaccard dissimilarity used
                               as follows:

                               For a pair of ontology terms A and B, if JaccardDistance(A,B) >= <uniqueness>, then we
                               keep both of these terms, and if a cell maps to both of them, it is duplicated in the
                               data (add another row, with the same expression, but its label will be different).
                               Otherwise, the sets of cells that map to these two terms overlap too much, and are too
                               similar, so we randomly pick one of them to use.

                               [default: 0.7]
"""

import json
#import pdb; pdb.set_trace()
from collections import defaultdict
from typing import List, Tuple, NamedTuple

import mygene
import numpy as np
import pandas as pd
from docopt import docopt
from scipy.spatial.distance import jaccard


def get_accessions_from_accessionSeries(accessionSeries_list):
    accessions = []
    for accessionSeries in accessionSeries_list:
        accession = int(accessionSeries.split('_')[0])
        accessions.append(accession)
    print("Bincounts of each accession #:")
    print(np.bincount(accessions))
    return np.array(accessions)


def convert_entrez_to_symbol(entrezIDs):
    mg = mygene.MyGeneInfo()
    result = mg.getgenes(entrezIDs, fields='symbol', species='mouse')
    symbols = [d['symbol'].lower() for d in result]
    return symbols


class DataFrameConstructionPOD(NamedTuple):
    true_id_index: List[str]
    index: List[str]
    expression_vectors: List[np.ndarray]
    labels: List[str]


def assign_terms_helper(pod: DataFrameConstructionPOD, all_terms: List[str], already_assigned_terms: List[int],
                        rpkm_df: pd.DataFrame, mapping_mat: np.ndarray, term_idx: int):
    if term_idx in already_assigned_terms:
        return
    term_str = all_terms[term_idx].split()[0]
    already_assigned_terms.append(term_idx)
    cell_selection_vector = mapping_mat[:, term_idx]
    num_to_add = np.sum(cell_selection_vector)
    pod.expression_vectors.extend(rpkm_df.loc[cell_selection_vector])
    old_index_values = rpkm_df.index[cell_selection_vector]
    new_index_values = [id + '_' + term_str for id in old_index_values]
    pod.index.extend(new_index_values)
    pod.true_id_index.extend(old_index_values)
    pod.labels.extend([all_terms[term_idx]] * num_to_add)


def assign_terms(rpkm_df: pd.DataFrame, mapping_mat: np.ndarray, terms, min_distance: float):
    pod = DataFrameConstructionPOD([], [], [], [])
    # Lists of indices into the 'terms' list which indicate which terms have been deleted, or have already
    # been assigned (added to the new rpkm DataFrame)
    # deleted_terms = []
    # assigned_terms = []
    num_terms = mapping_mat.shape[1]
    jaccard_distances = []
    for i in range(num_terms):
        for j in range(i + 1, num_terms):
            # if i in deleted_terms or j in deleted_terms:
            #     continue

            dist = jaccard(mapping_mat[:, i], mapping_mat[:, j])
            jaccard_distances.append(((terms[i], terms[j]), dist))
            # if dist >= min_distance:
            #     assign_terms_helper(pod, terms, assigned_terms, rpkm_df, mapping_mat, i)
            #     assign_terms_helper(pod, terms, assigned_terms, rpkm_df, mapping_mat, j)
            # else:
            #     assign_terms_helper(pod, terms, assigned_terms, rpkm_df, mapping_mat, i)
            #     deleted_terms.append(j)  # Choosing to delete the second term if they overlap too much
    # assigned_df = pd.DataFrame(data=np.asarray(pod.expression_vectors), columns=rpkm_df.columns, index=pod.index)
    # return assigned_df, pod.true_id_index, pod.labels
    sorted_distances = sorted(jaccard_distances, key=lambda x: x[1])
    print("Sorted Distances (increasing Jaccard distances) (first 50")
    for i in range(50):
        print(sorted_distances[i])    


def build_mapping_mat(cells, concise_mappings: dict) -> Tuple[np.ndarray, List[str]]:
    from itertools import chain
    terms = list(set(chain.from_iterable(concise_mappings.values()))) # All possible ontology terms
    terms = sorted(terms)
    num_cells = len(cells)
    num_terms = len(terms)
    mapping_mat = np.zeros((num_cells, num_terms))
    for i in range(num_cells):
        for j in range(num_terms):
            if terms[j] in concise_mappings[cells[i]]:
                mapping_mat[i, j] = 1
    return mapping_mat, terms


def load_cell_to_ontology_mapping(cells, ontology_mapping):
    empty_count = 0
    mappings = {}
    for cell in cells:
        terms_for_cell = ontology_mapping[cell]
        if len(terms_for_cell) == 0:
            empty_count += 1
        mappings[cell] = terms_for_cell
    print("Num cells with empty mappings: ", empty_count)
    return mappings


def analyze_cell_to_ontology_mapping(mappings):
    num_terms_mapped_to_l = []
    term_counts_d = defaultdict(int)
    for terms_for_cell in mappings.values():
        num_terms_mapped_to_l.append(len(terms_for_cell))
        for term in terms_for_cell:
            term_counts_d[term] += 1
    print("\nBincount of number of mapped terms for each cell:")
    print(np.bincount(num_terms_mapped_to_l))
    print("\nSorted list of terms by number of cells mapping to them (may overlap:")
    sorted_terms = sorted(term_counts_d.items(), key=lambda item: item[1], reverse=True)
    for term in sorted_terms:
        print(term)
    return term_counts_d


def filter_cell_to_ontology_terms(mappings, term_counts_d):
    terms_to_ignore = set()
    for term, count in term_counts_d.items():
        if count < 75 or 'NCBITaxon' in term or 'PR:' in term or 'PATO:' in term or 'GO:' in term or 'CLO:' in term:
            terms_to_ignore.add(term)
    # Terms that just don't seem that useful, or had too much overlap with another term
    terms_to_ignore.add('UBERON:0000006 islet of Langerhans')
    terms_to_ignore.add('CL:0000639 basophil cell of pars distalis of adenohypophysis')
    terms_to_ignore.add('CL:0000557 granulocyte monocyte progenitor cell')
    terms_to_ignore.add('UBERON:0001068 skin of back')
    terms_to_ignore.add('CL:0000034 stem cell')
    terms_to_ignore.add('CL:0000048 multi fate stem cell')
    terms_to_ignore.add('UBERON:0000178 blood')
    terms_to_ignore.add('UBERON:0001135 smooth muscle tissue')
    terms_to_ignore.add('UBERON:0001630 muscle organ')
    terms_to_ignore.add('CL:0000000 cell')
    terms_to_ignore.add('CL:0000080 circulating cell')

    # Clean the mappings
    for cell in mappings.keys():
        terms = mappings[cell]
        mappings[cell] = [term for term in terms if term not in terms_to_ignore]
    return mappings


def write_data_to_h5(name, df, labels, true_cell_ids=None, gene_symbols=None):
    print(df.shape)
    if not gene_symbols:
        gene_symbols = convert_entrez_to_symbol(df.columns)
    # Add these as Datasets in the hdf5 file
    gene_symbols_series = pd.Series(data=gene_symbols, index=df.columns)
    accessions = get_accessions_from_accessionSeries(df.index)
    accessions_series = pd.Series(data=accessions, index=df.index)
    labels_series = pd.Series(data=labels, index=df.index)

    new_h5_store = pd.HDFStore(name)
    new_h5_store['accessions'] = accessions_series
    new_h5_store['gene_symbols'] = gene_symbols_series
    new_h5_store['rpkm'] = df
    new_h5_store['labels'] = labels_series

    if true_cell_ids:
        new_h5_store['true_ids'] = pd.Series(data=true_cell_ids, index=df.index)
    new_h5_store.close()

    return gene_symbols


if __name__ == '__main__':
    args = docopt(__doc__, version='data_prep 0.1')

    # Open the hdf5 file that needs to be prepped (supplied as argument to this script)
    h5_store = pd.HDFStore(args['<expression_h5>'])
    print("loaded h5 file")
    rpkm_df = h5_store['rpkm']
    h5_store.close()
    print(rpkm_df.shape)
    rpkm_df.fillna(0, inplace=True)
    with open(args['<ontology_mappings_json>'], 'r') as f:
        cell_to_terms = json.load(f)
    mappings = load_cell_to_ontology_mapping(rpkm_df.index, cell_to_terms)
    # Analyze the mappings
    print("\n\nBEFORE FILTERING")
    term_counts_d = analyze_cell_to_ontology_mapping(mappings)
    # Filter the mappings
    mappings = filter_cell_to_ontology_terms(mappings, term_counts_d)
    print("\n\nAFTER FILTERING")
    analyze_cell_to_ontology_mapping(mappings)

    mapping_mat, terms = build_mapping_mat(rpkm_df.index, mappings)
    #assigned_rpkm_df, true_id_index, labels = assign_terms(rpkm_df, mapping_mat, terms, args['--uniq_lvl'])
    assign_terms(rpkm_df, mapping_mat, terms, args['--uniq_lvl'])
    import sys
    sys.exit()

    # # For now, let's keep only the cells that map to a single term
    # print("\n\nSelecting cells that map to a single term")
    # selected_cells = []
    # selected_labels = []
    # for key, value in mappings.items():
    #     if len(value) == 1:
    #         selected_cells.append(key)
    #         selected_labels.append(value[0])
    # selected_rpkm_df = rpkm_df.loc[selected_cells]
    # print("\nSelected cells expression matrix shape: ", selected_rpkm_df.shape)
    # unique_labels, counts = np.unique(selected_labels, return_counts=True)
    # print("\nThe selected labels and their counts (no overlap):")
    # print(unique_labels)
    # print(counts)

    # accessions = get_accessions_from_accessionSeries(selected_rpkm_df.index)
    accessions = get_accessions_from_accessionSeries(assigned_rpkm_df.index)
    print("converted to Accession #s")

    # Find cell types that exist in more than one Accession
    # Of these, pick the accession with the median number of cells of that cell type,
    # hold it out for the query set
    query_accn_for_label = {}

    label_to_accessions_d = defaultdict(lambda: defaultdict(int))
    for accession, label in zip(accessions, labels):
        label_to_accessions_d[label][accession] += 1

    print("\nAccessions for each cell type:")
    for label, accession_counts_d in label_to_accessions_d.items():
        print(label)
        accns_for_label = []
        accns_for_label_counts = []
        print("\t<acsn>: <count>")
        for accession, count in accession_counts_d.items():
            print("\t", accession, ": ", count)
            accns_for_label.append(accession)
            accns_for_label_counts.append(count)
        if len(accession_counts_d.keys()) >= 2:
            # Find accession with median number of cells of this type:
            sorted_indices = np.argsort(accns_for_label_counts)
            if len(sorted_indices == 2):
                median_idx = sorted_indices[0]
            else:
                median_idx = sorted_indices[len(accns_for_label_counts) // 2]
            query_accn = accns_for_label[median_idx]
            print("\tQuery accn: ", query_accn)
            query_accn_for_label[label] = query_accn
    # Split the dataset
    traindb_ids = []
    traindb_true_ids = []
    traindb_labels = []

    query_ids = []
    query_true_ids = []
    query_labels = []
    for cell_id, true_id, label, accession in zip(assigned_rpkm_df.index, true_id_index, labels, accessions):
        if label in query_accn_for_label and accession == query_accn_for_label[label]:
            query_ids.append(cell_id)
            query_true_ids.append(true_id)
            query_labels.append(label)
        else:
            traindb_ids.append(cell_id)
            traindb_true_ids.append(true_id)
            traindb_labels.append(label)

    gene_symbols = write_data_to_h5('selected_data.h5', assigned_rpkm_df, labels, true_id_index)
    write_data_to_h5('traindb_data.h5', assigned_rpkm_df.loc[traindb_ids], traindb_labels, traindb_true_ids, gene_symbols)
    write_data_to_h5('query_data.h5', assigned_rpkm_df.loc[query_ids], query_labels, query_true_ids, gene_symbols)
    print("Saved new h5 files")
