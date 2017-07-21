# import pdb; pdb.set_trace()
import pickle
import sys
import json
from collections import defaultdict

import pandas as pd
import numpy as np
import mygene

import ontology

# Conversion Dictionaries
#ENTREZ_TO_SYMBOL = 'entrez_to_mgi.pickle'
ENTREZ_TO_SYMBOL = 'data/mouse_data_20170718-133439_3623_cells/MGI_EntrezGene_20170720.rpt'
EXP_TO_TERM = 'data/mouse_data_20170718-133439_3623_cells/specific_experiment_term_mapping.json'

# Ontology files
ONTOLOGY = 'data/mouse_data_20170718-133439_3623_cells/ontology.pickle'

def fix_bad_entrezIDs(entrezIDs):
    to_fix = {
        19715: 100043034,
        113867: 100504631,
        212684: 100503041,
        215413: 70853,
        216164: 100503659,
        
        
        100042485: 105244994,
        113867: 100504631,
        19715: 100043034,
        212684: 100503041,
        215413: 70853,
        216164: 100503659,
        238944: 674895,
        240945: 269152,
        244425: 50768,
        319721: 78787,
        320169: 69398,
        320835: 14407,
        328644: 328643,
        347709: 347710,
        382643: 100041194,
        385605: 74478,
        406236: 232341,
        434446: 100502861
    }
    # fixed = []
    # for id in entrezIDs:
    #     if id in to_fix.keys():
    #         id = to_fix[id]
    #     fixed.append[id]
    # return fixed
    return [to_fix[id] if id in to_fix.keys() else id for id in entrezIDs]

def load_ontology(file):
    with open(file, 'rb') as f:
        ont = pickle.load(f)
    return ont

def get_terms_for_exp(exp_ids, mapping):
    return []

def get_term_obj_from_str(str):
    return ontology.Term(str.split()[0])

def load_entrez_to_mgi_mapping(filename):
    mapping = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        stripped = line.rstrip('\n').lower().split('\t')
        mgi_accession = stripped[0]
        entrezID = stripped[8]
        status = stripped[2]
        symbol = stripped[1]
        if status == 'w':
            # withdrawn
            continue
        if entrezID == '':
            print("Warning: no EntrezID for non-withdrawn MGI accession: ", mgi_accession, ", symbol: ", symbol)
            continue
        mapping[int(entrezID)] = symbol
    return mapping

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
        
    
if __name__ == '__main__':
    # Open the hdf5 file that needs to be prepped (supplied as argument to this script)
    h5_store = pd.HDFStore(sys.argv[1])
    print("loaded h5 file")
    rpkm_df = h5_store['rpkm']
    #experimentIDs = convert_accessionSeries_to_experimentID(rpkm_df.index)
    accessions = get_accessions_from_accessionSeries(rpkm_df.index)
    print("converted to experimentIDs")
    #entrez_to_symb_dict = load_entrez_to_mgi_mapping(ENTREZ_TO_SYMBOL)
    #entrezIDs = fix_bad_entrezIDs(rpkm_df.columns)
    entrezIDs = rpkm_df.columns
    #geneSymbols = convert(entrezIDs, entrez_to_symb_dict)
    #geneSymbols = convert_entrez_to_symbol(entrezIDs)
    print("converted EntrezIDs to MGI symbols")
    # Add these as Datasets in the hdf5 file

    # Analyze the mappings
    with open(EXP_TO_TERM, 'r') as f:
        cell_to_terms = json.load(f)
    num_terms = []
    mapping_counts = defaultdict(int)
    mappings = {}
    for cell in rpkm_df.index:
        terms_for_cell = cell_to_terms[cell]
        mappings[cell] = terms_for_cell
        num_terms.append(len(terms_for_cell))
        for term in terms_for_cell:
            mapping_counts[term] += 1
    print("Before filtering:")
    print("Bincount of number of mapped terms for each cell:")
    print(np.bincount(num_terms))
            
    terms_to_ignore = set()
    for term, count in mapping_counts.items():
        if count < 75 or 'NCBITaxon' in term or 'PR:' in term or 'PATO:' in term:
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
    # Second pass
    num_terms = []
    mapping_counts = defaultdict(int)
    for cell, terms in mappings.items():
        num_terms.append(len(terms))
        for term in terms:
            mapping_counts[term] += 1
        
        
    print("Bincount of number of mapped terms for each cell:")
    print(np.bincount(num_terms))
    print("Sorted list of terms by number of cells mapping to them:")
    sorted_terms = sorted(mapping_counts.items(), key=lambda item: item[1], reverse=True)
    for term in sorted_terms:
        print(term)

    print()
    
    # print("cells with 2 mappings:")
    # with open('2_mappings.txt', 'w') as f:
    #     for cell, terms in mappings.items():
    #         if len(terms) == 2:
    #             f.write(cell + "\t" + str(terms) + "\n")
    #             print(cell, "\t", terms)

    # For now, let's keep only the cells that map to a single term
    selected_cells = [key for key,value in mappings.items() if len(value) == 1]
    cleaned_rpkm = rpkm_df.loc[selected_cells]
    print(cleaned_rpkm.shape)
