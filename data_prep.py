"""Data Analysis & Preparation Script (for scRNA Pipeline)

Usage:
    data_prep.py <expression_h5> <ontology_mappings_json> [--filter --term_distances --assign=<assn>]
    data_prep.py (-h | --help)
    data_prep.py --version

Options:
    --filter            Remove terms that have less than 75 cells mapping to them. Also remove terms that are not part
                        of the main Cell Ontology DAG (i.e. NCBITaxon terms). Also remove some other, pre-determined
                        specific nodes (see the code for this list).
    --term_distances    Print the Jaccard distances between pairs of terms. The Jaccard distance is calculated between
                        the two sets of cells that map to the two terms.
    --assign=<assn>     Assign labels for the cells and split into a query set and a train/database set. The type of
                        assignment is dictated by the value of <assn>, which can be either:
                        - 'all'     Use all ontology terms that a cell maps to. For every term it maps to, create a
                                    copy of the cell's expression, with that term as the label.
                        - 'unique'  Use only cells that map to a single ontology term. Throw out cells that map to
                                    multiple terms.
"""

import json
# import pdb; pdb.set_trace()
from collections import defaultdict, namedtuple
from typing import List, Tuple

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


DataFrameConstructionPOD = namedtuple('DataFrameConstructionPOD', ['true_id_index', 'index', 'expression_vectors', 'labels'])

def assign_all_terms(all_terms: List[str],
                     rpkm_df: pd.DataFrame, mapping_mat: np.ndarray) -> DataFrameConstructionPOD:
    true_id_index = []
    index = []
    expression_vectors = []
    labels = []
    for term_idx in range(mapping_mat.shape[1]):
        term_str = all_terms[term_idx].split()[0]
        cell_selection_vector = mapping_mat[:, term_idx]
        num_to_add = np.sum(cell_selection_vector)
        old_index_values = rpkm_df.index[cell_selection_vector]
        expression_vectors.extend(rpkm_df.loc[old_index_values].values)
        new_index_values = [cell_id + '_' + term_str for cell_id in old_index_values]
        index.extend(new_index_values)
        true_id_index.extend(old_index_values)
        labels.extend([all_terms[term_idx]] * num_to_add)
    expression_vectors = np.asarray(expression_vectors)
    pod = DataFrameConstructionPOD(true_id_index, index, expression_vectors, labels)
    return pod


def assign_unique_terms(mappings, rpkm_df):
    pod = DataFrameConstructionPOD([], [], [], [])
    selected_cells = []
    selected_labels = []
    for key, value in mappings.items():
        if len(value) == 1:
            selected_cells.append(key)
            selected_labels.append(value[0])
    expression_vectors = rpkm_df.loc[selected_cells].values
    index = rpkm_df.loc[selected_cells].index
    true_id_index = index
    labels = selected_labels
    pod = DataFrameConstructionPOD(true_id_index, index, expression_vectors, labels)
    print("\nThe selected labels and their counts (no overlap):")
    unique_labels, counts = np.unique(selected_labels, return_counts=True)
    for l, c in zip(unique_labels, counts):
        print("\t", l, "\t", c)
    print()
    return pod

def calculate_term_distances(mapping_mat: np.ndarray, terms):
    num_terms = mapping_mat.shape[1]
    jaccard_distances = []
    for i in range(num_terms):
        for j in range(i + 1, num_terms):
            dist = jaccard(mapping_mat[:, i], mapping_mat[:, j])
            jaccard_distances.append(((terms[i], terms[j]), dist))
    sorted_distances = sorted(jaccard_distances, key=lambda x: x[1])
    print("\nSorted Distances (increasing Jaccard distances) (first 50)")
    for i in range(50):
        print(sorted_distances[i])    


def build_mapping_mat(cells, concise_mappings: dict) -> Tuple[np.ndarray, List[str]]:
    from itertools import chain
    terms = list(set(chain.from_iterable(concise_mappings.values()))) # All possible ontology terms
    terms = sorted(terms)
    num_cells = len(cells)
    num_terms = len(terms)
    mapping_mat = np.zeros((num_cells, num_terms), dtype=np.bool_)
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
    print("\nSorted list of terms by number of cells mapping to them (may overlap):")
    sorted_terms = sorted(term_counts_d.items(), key=lambda item: item[1], reverse=True)
    for term in sorted_terms:
        print(term)
    return term_counts_d


def filter_cell_to_ontology_terms(mappings, term_counts_d):
    terms_to_ignore = set()
    for term, count in term_counts_d.items():
        if count < 75 or 'NCBITaxon' in term or 'PR:' in term or 'PATO:' in term or 'GO:' in term or 'CLO:' in term:
            terms_to_ignore.add(term)
    # Terms that just don't seem that useful, or had too much overlap with another term that was more useful
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

    if true_cell_ids is not None:
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

    if args['--filter']:
        # Analyze the mappings
        print("\n\nBEFORE FILTERING")
        term_counts_d = analyze_cell_to_ontology_mapping(mappings)
        # Filter the mappings
        mappings = filter_cell_to_ontology_terms(mappings, term_counts_d)
        print("\n\nAFTER FILTERING")
        analyze_cell_to_ontology_mapping(mappings)
    mapping_mat = None
    if args['--assign'] == 'all' or args['--term_distances']:
        # These options require a mapping matrix
        mapping_mat, terms = build_mapping_mat(rpkm_df.index, mappings)
        #assigned_rpkm_df, true_id_index, labels = assign_terms(rpkm_df, mapping_mat, terms, args['--uniq_lvl'])
    if args['--term_distances']:
        calculate_term_distances(mapping_mat, terms)
    if args['--assign']:
        if args['--assign'] == 'all':
            pod = assign_all_terms(terms, rpkm_df, mapping_mat)
        elif args['--assign'] == 'unique':
            pod = assign_unique_terms(mappings, rpkm_df)
        assigned_rpkm_df = pd.DataFrame(data=pod.expression_vectors, columns=rpkm_df.columns, index=pod.index)
        accessions = get_accessions_from_accessionSeries(assigned_rpkm_df.index)
        # Find cell types that exist in more than one Accession
        # Of these, pick the accession with the median number of cells of that cell type,
        # hold it out for the query set
        query_accn_for_label = {}

        label_to_accessions_d = defaultdict(lambda: defaultdict(int))
        for accession, label in zip(accessions, pod.labels):
            label_to_accessions_d[label][accession] += 1

        # For Differential Expression, for a node, we only consider cells from the same study, and require
        # the study to have at least 100 cells (currently). Keep track of how many nodes will satisfy this.
        de_nodes = set()
        
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
                if count >= 75:
                    de_nodes.add(label)
            if len(accession_counts_d.keys()) >= 2:
                # Find accession with median number of cells of this type:
                sorted_indices = np.argsort(accns_for_label_counts)
                if len(sorted_indices) == 2:
                    median_idx = sorted_indices[0]
                else:
                    median_idx = sorted_indices[len(accns_for_label_counts) // 2]
                query_accn = accns_for_label[median_idx]
                print("\tQuery accn: ", query_accn)
                query_accn_for_label[label] = query_accn

        # Print out the query cell types
        print("Num query cell types: ", len(query_accn_for_label))
        for cell_type in query_accn_for_label.keys():
            print("\t", cell_type)

        # Print out nodes that we can do DE analysis for
        print("Num nodes for DE: ", len(de_nodes))
        for n in de_nodes:
            print("\t", n)
        
        # Split the dataset
        traindb_ids = []
        traindb_true_ids = []
        traindb_labels = []

        query_ids = []
        query_true_ids = []
        query_labels = []
        for cell_id, true_id, label, accession in zip(assigned_rpkm_df.index, pod.true_id_index, pod.labels, accessions):
            if label in query_accn_for_label and accession == query_accn_for_label[label]:
                query_ids.append(cell_id)
                query_true_ids.append(true_id)
                query_labels.append(label)
            else:
                traindb_ids.append(cell_id)
                traindb_true_ids.append(true_id)
                traindb_labels.append(label)

        gene_symbols = write_data_to_h5('selected_data.h5', assigned_rpkm_df, pod.labels, pod.true_id_index)
        write_data_to_h5('traindb_data.h5', assigned_rpkm_df.loc[traindb_ids], traindb_labels, traindb_true_ids,
                         gene_symbols)
        write_data_to_h5('query_data.h5', assigned_rpkm_df.loc[query_ids], query_labels, query_true_ids,
                         gene_symbols)
        print("Saved new h5 files")
    print("done")

