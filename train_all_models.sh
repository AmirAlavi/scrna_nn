#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32Gb
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user=aalavi@cs.cmu.edu
#SBATCH -o train_all_models.%N.%j.out
#SBATCH -e train_all_models.%N.%j.err

module load cuda-8.0

CUDNN_ROOT=$HOME/cudnn/cuda
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDNN_ROOT/lib64:$LIBRARY_PATH

# PCA BASELINES
python scrna.py train --pca=1136 --sn --out=models/mouse_genes_7_18_17_pca_1136
python scrna.py train --pca=100 --sn --out=models/mouse_genes_7_18_17_pca_100

# NON-SIAMESE MODELS
####################
# Dense 1136
python scrna.py train --nn=dense 1136 --sn --out=models/mouse_genes_7_18_17_dense_1136
# Dense 1136 100
python scrna.py train --nn=dense 1136 100 --sn --out=models/mouse_genes_7_18_17_dense_1136_100

# PPITF 1036+100
python scrna.py train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=models/mouse_genes_7_18_17_ppitf_1036.100
# PPITF 1036+100 100
python scrna.py train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=models/mouse_genes_7_18_17_ppitf_1036.100_100

# FlatGO 300+100
python scrna.py train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=models/mouse_genes_7_18_17_flatGO_300.100
# FlatGO 300+100 100
python scrna.py train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=models/mouse_genes_7_18_17_flatGO_300.100_100

# SIAMESE MODELS
################
# Dense 1136
python scrna.py train --nn=dense 1136 --sn --siamese --out=models/mouse_genes_7_18_17_siam_dense_1136
# Dense 1136 100
python scrna.py train --nn=dense 1136 100 --sn --siamese --out=models/mouse_genes_7_18_17_siam_dense_1136_100

# PPITF 1036+100
python scrna.py train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=models/mouse_genes_7_18_17_siam_ppitf_1036.100
# PPITF 1036+100 100
python scrna.py train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=models/mouse_genes_7_18_17_siam_ppitf_1036.100_100

# FlatGO 300+100
python scrna.py train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=models/mouse_genes_7_18_17_siam_flatGO_300.100
# FlatGO 300+100 100
python scrna.py train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=models/mouse_genes_7_18_17_siam_flatGO_300.100_100
