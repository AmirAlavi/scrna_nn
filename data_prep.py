#import pdb; pdb.set_trace()
import pickle
import sys
import json
from collections import defaultdict

import pandas as pd
import numpy as np
import mygene

import ontology

# Conversion Dictionaries
EXP_TO_TERM = 'data/mouse_data_20170718-133439_3623_cells/specific_experiment_term_mapping.json'

# Ontology files
ONTOLOGY = 'data/mouse_data_20170718-133439_3623_cells/ontology.pickle'

def load_ontology(file):
    with open(file, 'rb') as f:
        ont = pickle.load(f)
    return ont

def get_terms_for_exp(exp_ids, mapping):
    return []

def get_term_obj_from_str(str):
    return ontology.Term(str.split()[0])

def convert(to_convert, conversion_dict):
    converted = []
    unconverted = []
    for term in to_convert:
        if term in conversion_dict:
            converted.append(conversion_dict[term])
        else:
            unconverted.append(term)
    num_failed = len(unconverted)
    if num_failed > 0:
        raise LookupError("Unable to convert " + str(num_failed) + " terms " + str(unconverted))
    return converted


def get_accessions_from_accessionSeries(accessionSeries_list):
    accessions = []
    for accessionSeries in accessionSeries_list:
        accession = int(accessionSeries.split('_')[0])
        accessions.append(accession)
    print("Bincounts of each accession #:")
    print(np.bincount(accessions))
    return np.array(accessions)

def convert_accessionSeries_to_experimentID(accessionSeries_list):
    """Probably no use to this function, since accession numbers are already unique, and are
    globally unique, no need to create our own unique ID numbers for experiments.
    """
    accessions_seen = []
    experimentIDs = []
    for accessionSeries in accessionSeries_list:
        accession = accessionSeries.split('_')[0]
        if accession not in accessions_seen:
            accessions_seen.append(accession)
        expID = accessions_seen.index(accession)
        experimentIDs.append(expID)
    print("Bincounts of each experimentID:")
    print(np.bincount(experimentIDs))
    return np.array(experimentIDs)

def convert_entrez_to_symbol(entrezIDs):
    mg = mygene.MyGeneInfo()
    result = mg.getgenes(entrezIDs, fields='symbol', species='mouse')
    symbols = [d['symbol'].lower() for d in result]
    return symbols

def load_cell_to_ontology_mapping(cells, ontology_mapping):
    mappings = {}
    for cell in rpkm_df.index:
        terms_for_cell = cell_to_terms[cell]
        mappings[cell] = terms_for_cell
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
        if count < 75 or 'NCBITaxon' in term or 'PR:' in term or 'PATO:' in term or 'GO:' in term:
            terms_to_ignore.add(term)
    # Terms that just don't seem that useful, or had too much overlap with another term
    terms_to_ignore.add('UBERON:0000006 islet of Langerhans')
    terms_to_ignore.add('CL:0000639 basophil cell of pars distalis of adenohypophysis')
    terms_to_ignore.add('CL:0000557 granulocyte monocyte progenitor cell')
    terms_to_ignore.add('UBERON:0001068 skin of back')
    terms_to_ignore.add('CL:0000034 stem cell')
    # Clean the mappings
    for cell in mappings.keys():
        terms = mappings[cell]
        mappings[cell] = [term for term in terms if term not in terms_to_ignore]
    return mappings

def write_data_to_h5(name, df, labels, gene_symbols=None):
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
    new_h5_store.close()    

    return gene_symbols
    

if __name__ == '__main__':
    # Open the hdf5 file that needs to be prepped (supplied as argument to this script)
    h5_store = pd.HDFStore(sys.argv[1])
    print("loaded h5 file")
    rpkm_df = h5_store['rpkm']
    with open(EXP_TO_TERM, 'r') as f:
        cell_to_terms = json.load(f)
    mappings = load_cell_to_ontology_mapping(rpkm_df.index, cell_to_terms)
    # Analyze the mappings
    print("\n\nBEFORE FILTERING")
    term_counts_d = analyze_cell_to_ontology_mapping(mappings)
    # Filter the mappings
    mappings = filter_cell_to_ontology_terms(mappings, term_counts_d)
    print("\n\nAFTER FILTERING")
    analyze_cell_to_ontology_mapping(mappings)

    # For now, let's keep only the cells that map to a single term
    print("\n\nSelecting cells that map to a single term")
    selected_cells = []
    selected_labels = []
    for key, value in mappings.items():
        if len(value) == 1:
            selected_cells.append(key)
            selected_labels.append(value[0])
    selected_rpkm_df = rpkm_df.loc[selected_cells]
    print("\nSelected cells expression matrix shape: ", selected_rpkm_df.shape)
    unique_labels, counts = np.unique(selected_labels, return_counts=True)
    print("\nThe selected labels and their counts (no overlap):")
    print(unique_labels)
    print(counts)

    accessions = get_accessions_from_accessionSeries(selected_rpkm_df.index)
    print("converted to Accession #s")

    # Find cell types that exist in more than one Accession

    
    label_to_accessions_d = defaultdict(lambda: defaultdict(int))
    for accession, label  in zip(accessions, selected_labels):
        label_to_accessions_d[label][accession] += 1

    train_label_set = set()
    test_label_set = set()
    print("\nAccessions for each cell type:")
    for label, accession_counts_d in label_to_accessions_d.items():
        print(label)
        if len(accession_counts_d.keys()) >= 2:
            test_label_set.add(label)
        else:
            train_label_set.add(label)
        print("\t<acsn>: <count>")
        for accession, count in accession_counts_d.items():
            print("\t", accession, ": ", count)
    # Split the dataset
    train_ids = []
    train_labels = []
    test_ids = []
    test_labels = []
    for cell_id, label in zip(selected_cells, selected_labels):
        if label in train_label_set:
            train_ids.append(cell_id)
            train_labels.append(label)
        elif label in test_label_set:
            test_ids.append(cell_id)
            test_labels.append(label)

    gene_symbols = write_data_to_h5('selected_data.h5', selected_rpkm_df, selected_labels)
    write_data_to_h5('train_data.h5', selected_rpkm_df.loc[train_ids], train_labels, gene_symbols)
    write_data_to_h5('test_data.h5', selected_rpkm_df.loc[test_ids], test_labels, gene_symbols)
            
    # train_rpkm_df = selected_rpkm_df.loc[train_set_ids]
    # retrieval_test_rpkm_df = selected_rpkm_df.loc[retrieval_test_set_ids]
    
    # entrezIDs = selected_rpkm_df.columns
    # geneSymbols = convert_entrez_to_symbol(entrezIDs)
    # print("converted EntrezIDs to MGI symbols")
    # # Add these as Datasets in the hdf5 file
    # geneSymbols_series = pd.Series(data=geneSymbols, index=selected_rpkm_df.columns)
    # accessions_series = pd.Series(data=accessions, index=selected_rpkm_df.index)
    # labels_series = pd.Series(data=selected_labels, index=selected_rpkm_df.index)

    # new_h5_store = pd.HDFStore('selected_expr_data.h5')
    # new_h5_store['accessions'] = accessions_series
    # new_h5_store['gene_symbols'] = geneSymbols_series
    # new_h5_store['rpkm'] = selected_rpkm_df
    # new_h5_store['labels'] = labels_series
    # new_h5_store.close()

    # train_h5_store = pd.HDFStore('train_expr_data.h5')
    # new_h5_store['accessions'] = 
    # new_h5_store['gene_symbols'] = geneSymbols_series
    # new_h5_store['rpkm'] = _rpkm_df
    # new_h5_store['labels'] = labels_series
    # new_h5_store.close()
    h5_store.close()

    # Finally, split up into train and test
    
    print("Saved new h5 files")
