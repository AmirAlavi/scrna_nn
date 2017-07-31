#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 50Gb
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user=aalavi@cs.cmu.edu
#SBATCH -o train_all_models.%N.%j.out
#SBATCH -e train_all_models.%N.%j.err

module load cuda-8.0

CUDNN_ROOT=$HOME/cudnn/cuda
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDNN_ROOT/lib64:$LIBRARY_PATH

#TRAIN_DATA=data/mouse_data_20170718-133439_3623_cells/traindb_data.h5
TRAIN_DATA=data/mouse_data_20170728-102617_5349_cells/traindb_data.h5
MODEL_PREFIX=models/mouse_genes_7_31_17_

# PCA BASELINES
python scrna.py train --pca=1136 --sn --out=${MODEL_PREFIX}pca_1136 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}pca_1136 > experiment_models.list
python scrna.py train --pca=100 --sn --out=${MODEL_PREFIX}pca_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}pca_100 >> experiment_models.list
python scrna.py train --pca=500 --sn --out=${MODEL_PREFIX}pca_500 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}pca_500 >> experiment_models.list

# NON-SIAMESE MODELS
####################
# Dense 1136
python scrna.py train --nn=dense 1136 --sn --out=${MODEL_PREFIX}dense_1136 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}dense_1136 >> experiment_models.list
# Dense 1136 100
python scrna.py train --nn=dense 1136 100 --sn --out=${MODEL_PREFIX}dense_1136_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}dense_1136_100 >> experiment_models.list
# Dense 1136 500
python scrna.py train --nn=dense 1136 500 --sn --out=${MODEL_PREFIX}dense_1136_500 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}dense_1136_500 >> experiment_models.list
# Dense 1136 500 100
python scrna.py train --nn=dense 1136 500 100 --sn --out=${MODEL_PREFIX}dense_1136_500_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}dense_1136_500_100 >> experiment_models.list

# PPITF 1036+100
python scrna.py train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}ppitf_1036.100 >> experiment_models.list
# PPITF 1036+100 100
python scrna.py train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}ppitf_1036.100_100 >> experiment_models.list
# PPITF 1036+100 500
python scrna.py train --nn=sparse 500 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100_500 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}ppitf_1036.100_500 >> experiment_models.list
# PPITF 1036+100 500 100
python scrna.py train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100_500_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}ppitf_1036.100_500_100 >> experiment_models.list

# FlatGO 300+100
python scrna.py train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}flatGO_300.100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}flatGO_300.100 >> experiment_models.list
# FlatGO 300+100 100
python scrna.py train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}flatGO_300.100_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}flatGO_300.100_100 >> experiment_models.list
# FlatGO 300+100 200
python scrna.py train --nn=sparse 200 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}flatGO_300.100_200 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}flatGO_300.100_200 >> experiment_models.list

# SIAMESE MODELS
################
# Dense 1136
python scrna.py train --nn=dense 1136 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_dense_1136 >> experiment_models.list
# Dense 1136 100
python scrna.py train --nn=dense 1136 100 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_dense_1136_100 >> experiment_models.list
# Dense 1136 500
python scrna.py train --nn=dense 1136 500 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136_500 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_dense_1136_500 >> experiment_models.list
# Dense 1136 500 100
python scrna.py train --nn=dense 1136 500 100 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136_500_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_dense_1136_500_100 >> experiment_models.list

# PPITF 1036+100
python scrna.py train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_ppitf_1036.100 >> experiment_models.list
# PPITF 1036+100 100
python scrna.py train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_ppitf_1036.100_100 >> experiment_models.list
# PPITF 1036+100 500
python scrna.py train --nn=sparse 500 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100_500 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_ppitf_1036.100_500 >> experiment_models.list
# PPITF 1036+100 500 100
python scrna.py train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100_500_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_ppitf_1036.100_500_100 >> experiment_models.list

# FlatGO 300+100
python scrna.py train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_flatGO_300.100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_flatGO_300.100 >> experiment_models.list
# FlatGO 300+100 100
python scrna.py train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_flatGO_300.100_100 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_flatGO_300.100_100 >> experiment_models.list
# FlatGO 300+100 200
python scrna.py train --nn=sparse 200 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_flatGO_300.100_200 --data=$TRAIN_DATA
echo ${MODEL_PREFIX}siam_flatGO_300.100_200 >> experiment_models.list
