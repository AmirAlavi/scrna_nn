#!/bin/bash

# Basic help message:
if [ $# -eq 0 ]; then
    printf "Usage:\n train_models.sh <model_name_prefix> <train_data> <train_session_name> <models:all/non-siamese/siamese> <email> <common_options...>\n"
    exit 1
fi

MODEL_PREFIX=$1
TRAIN_DATA=$2
SESSION_NAME=$3
MODELS=$4
EMAIL=$5
EXTRA_OPTS="${@:6}"

> ${SESSION_NAME}_commands.list
> ${SESSION_NAME}_models.list

if [ "$MODELS" == "all" -o "$MODELS" == "pca" ]; then
    # PCA BASELINES
    echo  scrna-nn train --pca=1136 --sn --out=${MODEL_PREFIX}pca_1136 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_1136 >> ${SESSION_NAME}_models.list
    echo  scrna-nn train --pca=500 --sn --out=${MODEL_PREFIX}pca_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_500 >> ${SESSION_NAME}_models.list
    echo  scrna-nn train --pca=200 --sn --out=${MODEL_PREFIX}pca_200 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_200 >> ${SESSION_NAME}_models.list
    echo  scrna-nn train --pca=100 --sn --out=${MODEL_PREFIX}pca_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_100 >> ${SESSION_NAME}_models.list
    echo  scrna-nn train --pca=50 --sn --out=${MODEL_PREFIX}pca_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_50 >> ${SESSION_NAME}_models.list
fi

if [ "$MODELS" == "all" -o "$MODELS" == "non-siamese" -o "$MODELS" == "neural-nets" ]; then
    # NON-SIAMESE MODELS
    ####################
    # Dense 1136
    echo  scrna-nn train --nn=dense 1136 --sn --out=${MODEL_PREFIX}dense_1136 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136 >> ${SESSION_NAME}_models.list
    # Dense 1136 100
    echo  scrna-nn train --nn=dense 1136 100 --sn --out=${MODEL_PREFIX}dense_1136_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136_100 >> ${SESSION_NAME}_models.list
    # Dense 1136 500
    echo  scrna-nn train --nn=dense 1136 500 --sn --out=${MODEL_PREFIX}dense_1136_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136_500 >> ${SESSION_NAME}_models.list
    # Dense 1136 500 100
    echo  scrna-nn train --nn=dense 1136 500 100 --sn --out=${MODEL_PREFIX}dense_1136_500_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136_500_100 >> ${SESSION_NAME}_models.list
    # Dense 1136 500 100 50
    echo  scrna-nn train --nn=dense 1136 500 100 50 --sn --out=${MODEL_PREFIX}dense_1136_500_100_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136_500_100_50 >> ${SESSION_NAME}_models.list
    
    
    # PPITF 1036+100
    echo  scrna-nn train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100 >> ${SESSION_NAME}_models.list
    # PPITF 1036+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100_100 >> ${SESSION_NAME}_models.list
    # PPITF 1036+100 500
    echo  scrna-nn train --nn=sparse 500 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100_500 >> ${SESSION_NAME}_models.list
    # PPITF 1036+100 500 100
    echo  scrna-nn train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100_500_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100_500_100 >> ${SESSION_NAME}_models.list
    # PPITF 1036+100 500 100
    echo  scrna-nn train --nn=sparse 500 100 50 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}ppitf_1036.100_500_100_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100_500_100_50 >> ${SESSION_NAME}_models.list
    
    # FlatGO 300+100
    echo  scrna-nn train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}flatGO_300.100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}flatGO_300.100 >> ${SESSION_NAME}_models.list
   # FlatGO 300+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}flatGO_300.100_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}flatGO_300.100_100 >> ${SESSION_NAME}_models.list
    # FlatGO 300+100 200
    echo  scrna-nn train --nn=sparse 200 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}flatGO_300.100_200 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
   echo ${MODEL_PREFIX}flatGO_300.100_200 >> ${SESSION_NAME}_models.list
   # FlatGO 300+100 200 50
   echo  scrna-nn train --nn=sparse 200 50 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}flatGO_300.100_200_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
   echo ${MODEL_PREFIX}flatGO_300.100_200_50 >> ${SESSION_NAME}_models.list
   
   # GO Levels 4-2
   echo  scrna-nn train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31 --sn --out=${MODEL_PREFIX}GO_lvls_4_3_2.31 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
   echo ${MODEL_PREFIX}GO_lvls_4_3_2.31 >> ${SESSION_NAME}_models.list
   
   # Combined FlatGO + PPITF
   echo  scrna-nn train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=0 --sn --out=${MODEL_PREFIX}comb_flatGO_ppitf --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
   echo ${MODEL_PREFIX}comb_flatGO_ppitf >> ${SESSION_NAME}_models.list
   # Combined FlatGO + PPITF + Dense
   echo  scrna-nn train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=100 --sn --out=${MODEL_PREFIX}comb_flatGO_ppitf_dense --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
   echo ${MODEL_PREFIX}comb_flatGO_ppitf_dense >> ${SESSION_NAME}_models.list
fi

if [ "$MODELS" == "all" -o "$MODELS" == "siamese" -o "$MODELS" == "neural-nets" ]; then
    # SIAMESE MODELS
    ################
    # Dense 1136
    echo  scrna-nn train --nn=dense 1136 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136 >> ${SESSION_NAME}_models.list
    # Dense 1136 100
    echo  scrna-nn train --nn=dense 1136 100 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136_100 >> ${SESSION_NAME}_models.list
    # Dense 1136 500
    echo  scrna-nn train --nn=dense 1136 500 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136_500 >> ${SESSION_NAME}_models.list
    # Dense 1136 500 100
    echo  scrna-nn train --nn=dense 1136 500 100 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136_500_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136_500_100 >> ${SESSION_NAME}_models.list
    # Dense 1136 500 100 50
    echo  scrna-nn train --nn=dense 1136 500 100 50 --sn --siamese --out=${MODEL_PREFIX}siam_dense_1136_500_100_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136_500_100_50 >> ${SESSION_NAME}_models.list
    
    # PPITF 1036+100
    echo  scrna-nn train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100 >> ${SESSION_NAME}_models.list
    # PPITF 1036+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100_100 >> ${SESSION_NAME}_models.list
    # PPITF 1036+100 500
    echo  scrna-nn train --nn=sparse 500 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100_500 >> ${SESSION_NAME}_models.list
    # PPITF 1036+100 500 100
    echo  scrna-nn train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100_500_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100_500_100 >> ${SESSION_NAME}_models.list
    # PPITF 1036+100 500 100 50
    echo  scrna-nn train --nn=sparse 500 100 50 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_ppitf_1036.100_500_100_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100_500_100_50 >> ${SESSION_NAME}_models.list
    
    # FlatGO 300+100
    echo  scrna-nn train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_flatGO_300.100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100 >> ${SESSION_NAME}_models.list
    # FlatGO 300+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_flatGO_300.100_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100_100 >> ${SESSION_NAME}_models.list
    # FlatGO 300+100 200
    echo  scrna-nn train --nn=sparse 200 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_flatGO_300.100_200 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100_200 >> ${SESSION_NAME}_models.list
    # FlatGO 300+100 200 50
    echo  scrna-nn train --nn=sparse 200 50 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_flatGO_300.100_200_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100_200_50 >> ${SESSION_NAME}_models.list
    
    # GO Levels 4-2
    echo  scrna-nn train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31 --sn --siamese --out=${MODEL_PREFIX}siam_GO_lvls_4_3_2.31 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_GO_lvls_4_3_2.31 >> ${SESSION_NAME}_models.list
    
    # Combined FlatGO + PPITF
    echo  scrna-nn train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=0 --sn --siamese --out=${MODEL_PREFIX}siam_comb_flatGO_ppitf --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_comb_flatGO_ppitf >> ${SESSION_NAME}_models.list
    # Combined FlatGO + PPITF + Dense
    echo  scrna-nn train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=100 --sn --siamese --out=${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense >> ${SESSION_NAME}_models.list    
fi

NUM_JOBS=$(wc -l ${SESSION_NAME}_commands.list | awk {'print $1'})
echo "$NUM_JOBS"
NUM_JOBS=$(($NUM_JOBS - 1))
echo "$NUM_JOBS"
sbatch --array=0-$NUM_JOBS --mail-user ${EMAIL} slurm/train_model_arr.sh ${SESSION_NAME}_commands.list
