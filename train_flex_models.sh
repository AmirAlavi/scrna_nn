#!/bin/bash

# Basic help message:
if [ $# -eq 0 ]; then
    printf "Usage:\n train_all_models.sh <model_name_prefix> <train_data>\n"
    exit 1
fi

TRAIN_DATA=$2
MODEL_PREFIX=$1

# SIAMESE MODELS
################
# Dense 1136
echo python scrna.py train --nn=dense 1136 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_dense_1136 --data=$TRAIN_DATA > train_commands.list
echo ${MODEL_PREFIX}siam_dense_1136 >> experiment_models.list
# Dense 1136 100
echo python scrna.py train --nn=dense 1136 100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_dense_1136_100 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_dense_1136_100 >> experiment_models.list
# Dense 1136 500
echo python scrna.py train --nn=dense 1136 500 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_dense_1136_500 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_dense_1136_500 >> experiment_models.list
# Dense 1136 500 100
echo python scrna.py train --nn=dense 1136 500 100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_dense_1136_500_100 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_dense_1136_500_100 >> experiment_models.list
# Dense 1136 500 100 50
echo python scrna.py train --nn=dense 1136 500 100 50 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_dense_1136_500_100_50 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_dense_1136_500_100_50 >> experiment_models.list

# PPITF 1036+100
echo python scrna.py train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_ppitf_1036.100 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_ppitf_1036.100 >> experiment_models.list
# PPITF 1036+100 100
echo python scrna.py train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_ppitf_1036.100_100 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_ppitf_1036.100_100 >> experiment_models.list
# PPITF 1036+100 500
echo python scrna.py train --nn=sparse 500 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_ppitf_1036.100_500 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_ppitf_1036.100_500 >> experiment_models.list
# PPITF 1036+100 500 100
echo python scrna.py train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_ppitf_1036.100_500_100 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_ppitf_1036.100_500_100 >> experiment_models.list
# PPITF 1036+100 500 100 50
echo python scrna.py train --nn=sparse 500 100 50 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_ppitf_1036.100_500_100_50 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_ppitf_1036.100_500_100_50 >> experiment_models.list

# FlatGO 300+100
echo python scrna.py train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_flatGO_300.100 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_flatGO_300.100 >> experiment_models.list
# FlatGO 300+100 100
echo python scrna.py train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_flatGO_300.100_100 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_flatGO_300.100_100 >> experiment_models.list
# FlatGO 300+100 200
echo python scrna.py train --nn=sparse 200 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_flatGO_300.100_200 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_flatGO_300.100_200 >> experiment_models.list
# FlatGO 300+100 200 50
echo python scrna.py train --nn=sparse 200 50 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_flatGO_300.100_200_50 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_flatGO_300.100_200_50 >> experiment_models.list

# GO Levels 4-2
echo python scrna.py train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_GO_lvls_4_3_2.31 --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_GO_lvls_4_3_2.31 >> experiment_models.list

# Combined FlatGO + PPITF
echo python scrna.py train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=0 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_comb_flatGO_ppitf --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_comb_flatGO_ppitf >> experiment_models.list
# Combined FlatGO + PPITF + Dense
echo python scrna.py train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --flexibleLoss=dist_mat_by_strings.p --out=${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense --data=$TRAIN_DATA >> train_commands.list
echo ${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense >> experiment_models.list

NUM_JOBS=$(wc -l train_commands.list | awk {'print $1'})
echo "$NUM_JOBS"
NUM_JOBS=$(($NUM_JOBS - 1))
echo "$NUM_JOBS"
sbatch --array=0-$NUM_JOBS train_model_arr.sh
