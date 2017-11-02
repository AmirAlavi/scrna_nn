# arg1 = name of rpkm h5 file to convert to counts
# arg2 = name of h5 file that contains alignment metadata (must contain at least all of the samples in arg1 rpkm dataframe)
import pickle
import sys

import pandas as pd
import numpy as np
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# Load mapping of geneID to gene_length
with open('trees.pickle', 'rb') as f:
    trees = pickle.load(f)
gene_lengths = trees['gene_length'] # dict of geneID -> length in KB
gene_lengths.index = gene_lengths.index.map(int) # necessary because the index (geneIDs) was strings originally

# load rpkm dataframe
store = pd.HDFStore(sys.argv[1])
rpkm = store['rpkm']
# Also load other metadata that we might need later downstream
# Include these for the count matrix just like we did for rpkm
accessions = store['accessions']
gene_symbols = store['gene_symbols']
labels = store['labels']
store.close()

# Load mapping of sampleName to num_reads
store = pd.HDFStore(sys.argv[2])
alignment_meta = store['alignment_metadata']
store.close()
read_counts = alignment_meta['read_count'].loc[rpkm.index] # select only the values for the samples present in our rpkm matrix

# Sanity check that the orderings are what we expect
np.testing.assert_array_equal(rpkm.index, read_counts.index)
np.testing.assert_array_equal(rpkm.columns, gene_lengths.index)
print("passed assertions")
counts_mat = rpkm
counts_mat = counts_mat.mul(gene_lengths, axis=1)
counts_mat = counts_mat.mul(read_counts, axis=0)
counts_mat /= 1000000
counts_mat = counts_mat.round()
counts_mat = counts_mat.astype(int)
print(counts_mat.shape)
# new_store = pd.HDFStore('selected_counts.h5')
# new_store['counts'] = counts_mat
# new_store['accessions'] = accessions
# new_store['gene_symbols'] = gene_symbols
# new_store['labels'] = labels
# new_store.close()
print("converting to r objects")
counts_mat = pandas2ri.py2ri(counts_mat)
print("converted")
r.assign("counts.mat", counts_mat)
print("assigned")
r("save(counts.mat, file='counts_mat.gzip', compress=TRUE)")
print("saved")

print("converting to r objects")
accessions = pandas2ri.py2ri(accessions)
print("converted")
r.assign("accessions", accessions)
print("assigned")
r("save(accessions, file='accessions.gzip', compress=TRUE)")
print("saved")

print("converting to r objects")
gene_symbols = pandas2ri.py2ri(gene_symbols)
print("converted")
r.assign("gene.symbols", gene_symbols)
print("assigned")
r("save(gene.symbols, file='gene_symbols.gzip', compress=TRUE)")
print("saved")

print("converting to r objects")
labels = pandas2ri.py2ri(labels)
print("converted")
r.assign("labels", labels)
print("assigned")
r("save(labels, file='labels.gzip', compress=TRUE)")
print("saved")
print("done")
