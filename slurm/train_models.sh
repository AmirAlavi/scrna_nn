#!/bin/bash

# Basic help message:
if [ $# -eq 0 ]; then
    printf "Usage:\n train_models.sh <model_name_prefix> <train_data> <train_session_name> <models:all/non-siamese/siamese> <email> <use_scratch_T_or_F> <common_options...>\n"
    exit 1
fi

MODEL_PREFIX=$1
TRAIN_DATA=$2
SESSION_NAME=$3
MODELS=$4
EMAIL=$5
SCRATCH=$6
EXTRA_OPTS="${@:7}"

> ${SESSION_NAME}_commands.list
> ${SESSION_NAME}_models.list
> ${SESSION_NAME}_scratch_copy_commands.list

if [ "$SCRATCH" == "scratch:T" ]; then
    DO_SCRATCH=""
    SCRATCH_PREFIX="/scratch/aalavi/"
    mkdir -p ${MODEL_PREFIX}
elif [ "$SCRATCH" == "scratch:F" ]; then
    DO_SCRATCH=": " # noop used if you don't want to use scratch space
    SCRATCH_PREFIX=""
else
    printf "Scratch option must be either 'scratch:T' or 'scratch:F'!\n"
    exit 1
fi

if [ "$MODELS" == "all" -o "$MODELS" == "pca" ]; then
    # PCA BASELINES
    echo  scrna-nn train --pca=1136  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pca_1136 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_1136 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pca_1136 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    echo  scrna-nn train --pca=500  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pca_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_500 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pca_500 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    echo  scrna-nn train --pca=200  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pca_200 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_200 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pca_200 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    echo  scrna-nn train --pca=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pca_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pca_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    echo  scrna-nn train --pca=50  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pca_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pca_50 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pca_50 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
fi

if [ "$MODELS" == "all" -o "$MODELS" == "non-siamese" -o "$MODELS" == "neural-nets" ]; then
    # NON-SIAMESE MODELS
    ####################
    # Dense 1136
    echo  scrna-nn train --nn=dense 1136  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 100
    echo  scrna-nn train --nn=dense 1136 100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 500
    echo  scrna-nn train --nn=dense 1136 500  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136_500 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136_500 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 500 100
    echo  scrna-nn train --nn=dense 1136 500 100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136_500_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136_500_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136_500_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 500 100 50
    echo  scrna-nn train --nn=dense 1136 500 100 50  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136_500_100_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}dense_1136_500_100_50 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}dense_1136_500_100_50 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    # PPITF 1036+100
    echo  scrna-nn train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 500
    echo  scrna-nn train --nn=sparse 500 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100_500 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100_500 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 500 100
    echo  scrna-nn train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100_500_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100_500_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100_500_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 500 100 50
    echo  scrna-nn train --nn=sparse 500 100 50 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100_500_100_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}ppitf_1036.100_500_100_50 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}ppitf_1036.100_500_100_50 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    # FlatGO 300+100
    echo  scrna-nn train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}flatGO_300.100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}flatGO_300.100_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 200
    echo  scrna-nn train --nn=sparse 200 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100_200 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}flatGO_300.100_200 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100_200 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 200 100
    echo  scrna-nn train --nn=sparse 200 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100_200_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}flatGO_300.100_200_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100_200_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 200 50
    echo  scrna-nn train --nn=sparse 200 50 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100_200_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}flatGO_300.100_200_50 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}flatGO_300.100_200_50 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
   
    # GO Levels 4-2
    echo  scrna-nn train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}GO_lvls_4_3_2.31 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}GO_lvls_4_3_2.31 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}GO_lvls_4_3_2.31 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
   
    # Combined FlatGO + PPITF
    echo  scrna-nn train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=0  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}comb_flatGO_ppitf --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}comb_flatGO_ppitf >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}comb_flatGO_ppitf ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Combined FlatGO + PPITF 100
    echo  scrna-nn train --nn=flatGO_ppitf 100 --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=0  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}comb_flatGO_ppitf_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}comb_flatGO_ppitf_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}comb_flatGO_ppitf_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Combined FlatGO + PPITF + Dense
    echo  scrna-nn train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}comb_flatGO_ppitf_dense --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}comb_flatGO_ppitf_dense >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}comb_flatGO_ppitf_dense ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Combined FlatGO + PPITF + Dense 100
    echo  scrna-nn train --nn=flatGO_ppitf 100 --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}comb_flatGO_ppitf_dense_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}comb_flatGO_ppitf_dense_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}comb_flatGO_ppitf_dense_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
fi

if [ "$MODELS" == "pretrained" ]; then
    # # Dense 1136 100
    # echo  scrna-nn train --nn=dense 1136 100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_dense_1136_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/dense_1136_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_dense_1136_100 >> ${SESSION_NAME}_models.list
    # # Dense 1136 500 100
    # echo  scrna-nn train --nn=dense 1136 500 100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_dense_1136_500_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/dense_1136_500_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_dense_1136_500_100 >> ${SESSION_NAME}_models.list
    # # PPITF 1036+100 100
    # echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_ppitf_1036.100_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/ppitf_1136_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_ppitf_1036.100_100 >> ${SESSION_NAME}_models.list
    # # PPITF 1036+100 500 100
    # echo  scrna-nn train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_ppitf_1036.100_500_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/ppitf_1136_500_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_ppitf_1036.100_500_100 >> ${SESSION_NAME}_models.list
    # # GO Levels 4-2
    # echo  scrna-nn train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_GO_lvls_4_3_2.31 --data=$TRAIN_DATA --unsup_pt=pre_trained/GO_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_GO_lvls_4_3_2.31 >> ${SESSION_NAME}_models.list

    # Dense 1136 100
    echo  scrna-nn train --nn=dense 1136 100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_dense_1136_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/dense_1136_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_dense_1136_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_dense_1136_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 500 100
    echo  scrna-nn train --nn=dense 1136 500 100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_dense_1136_500_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/dense_1136_500_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_dense_1136_500_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_dense_1136_500_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_ppitf_1036.100_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/ppitf_1136_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_ppitf_1036.100_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_ppitf_1036.100_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 500 100
    echo  scrna-nn train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_ppitf_1036.100_500_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/ppitf_1136_500_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_ppitf_1036.100_500_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_ppitf_1036.100_500_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_flatGO_300.100_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/flatGO_400_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_flatGO_300.100_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_flatGO_300.100_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 200 100
    echo  scrna-nn train --nn=sparse 200 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_flatGO_300.100_200_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/flatGO_400_200_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_flatGO_300.100_200_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_flatGO_300.100_200_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # GO Levels 4-2
    echo  scrna-nn train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31  --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_GO_lvls_4_3_2.31 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/GO_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_GO_lvls_4_3_2.31 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_GO_lvls_4_3_2.31 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
fi

if [ "$MODELS" == "siam-pretrained" ]; then
    # # Dense 1136 100
    # echo  scrna-nn train --nn=dense 1136 100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_dense_1136_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/dense_1136_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_siam_dense_1136_100 >> ${SESSION_NAME}_models.list
    # # Dense 1136 500 100
    # echo  scrna-nn train --nn=dense 1136 500 100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_dense_1136_500_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/dense_1136_500_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_siam_dense_1136_500_100 >> ${SESSION_NAME}_models.list
    # # PPITF 1036+100 100
    # echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_ppitf_1036.100_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/ppitf_1136_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_siam_ppitf_1036.100_100 >> ${SESSION_NAME}_models.list
    # # PPITF 1036+100 500 100
    # echo  scrna-nn train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_ppitf_1036.100_500_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/ppitf_1136_500_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_siam_ppitf_1036.100_500_100 >> ${SESSION_NAME}_models.list
    # # GO Levels 4-2
    # echo  scrna-nn train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_GO_lvls_4_3_2.31 --data=$TRAIN_DATA --unsup_pt=pre_trained/GO_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    # echo ${MODEL_PREFIX}pt_siam_GO_lvls_4_3_2.31 >> ${SESSION_NAME}_models.list

    # Dense 1136 100
    echo  scrna-nn train --nn=dense 1136 100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_dense_1136_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/dense_1136_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_siam_dense_1136_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_dense_1136_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 500 100
    echo  scrna-nn train --nn=dense 1136 500 100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_dense_1136_500_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/dense_1136_500_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_siam_dense_1136_500_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_dense_1136_500_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_ppitf_1036.100_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/ppitf_1136_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_siam_ppitf_1036.100_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_ppitf_1036.100_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 500 100
    echo  scrna-nn train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_ppitf_1036.100_500_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/ppitf_1136_500_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_siam_ppitf_1036.100_500_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_ppitf_1036.100_500_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_flatGO_300.100_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/flatGO_400_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_flatGO_300.100_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_flatGO_300.100_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 200 100
    echo  scrna-nn train --nn=sparse 200 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_flatGO_300.100_200_100 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/flatGO_400_200_100_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_flatGO_300.100_200_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_flatGO_300.100_200_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # GO Levels 4-2
    echo  scrna-nn train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_GO_lvls_4_3_2.31 --data=$TRAIN_DATA --unsup_pt=pre_trained/2018_01_28/GO_tanh_gn $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}pt_siam_GO_lvls_4_3_2.31 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}pt_siam_GO_lvls_4_3_2.31 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
fi

if [ "$MODELS" == "all" -o "$MODELS" == "siamese" -o "$MODELS" == "neural-nets" ]; then
    # SIAMESE MODELS
    ################
    # Dense 1136
    echo  scrna-nn train --nn=dense 1136  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 100
    echo  scrna-nn train --nn=dense 1136 100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 500
    echo  scrna-nn train --nn=dense 1136 500  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136_500 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136_500 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 500 100
    echo  scrna-nn train --nn=dense 1136 500 100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136_500_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136_500_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136_500_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Dense 1136 500 100 50
    echo  scrna-nn train --nn=dense 1136 500 100 50  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136_500_100_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_dense_1136_500_100_50 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_dense_1136_500_100_50 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    # PPITF 1036+100
    echo  scrna-nn train --nn=sparse --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 500
    echo  scrna-nn train --nn=sparse 500 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100_500 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100_500 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100_500 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 500 100
    echo  scrna-nn train --nn=sparse 500 100 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100_500_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100_500_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100_500_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # PPITF 1036+100 500 100 50
    echo  scrna-nn train --nn=sparse 500 100 50 --sparse_groupings=data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100_500_100_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_ppitf_1036.100_500_100_50 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_ppitf_1036.100_500_100_50 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    # FlatGO 300+100
    echo  scrna-nn train --nn=sparse --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 100
    echo  scrna-nn train --nn=sparse 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 200
    echo  scrna-nn train --nn=sparse 200 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100_200 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100_200 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100_200 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 200 100
    echo  scrna-nn train --nn=sparse 200 100 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100_200_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100_200_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100_200_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # FlatGO 300+100 200 50
    echo  scrna-nn train --nn=sparse 200 50 --sparse_groupings=data/flat_GO300_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100_200_50 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_flatGO_300.100_200_50 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_flatGO_300.100_200_50 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    # GO Levels 4-2
    echo  scrna-nn train --nn=GO --go_arch=data/GO_lvls_arch_2_to_4 --with_dense=31  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_GO_lvls_4_3_2.31 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_GO_lvls_4_3_2.31 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_GO_lvls_4_3_2.31 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    
    # Combined FlatGO + PPITF
    echo  scrna-nn train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=0  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_comb_flatGO_ppitf --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_comb_flatGO_ppitf >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_comb_flatGO_ppitf ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Combined FlatGO + PPITF 100
    echo  scrna-nn train --nn=flatGO_ppitf 100 --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=0  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_comb_flatGO_ppitf_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_comb_flatGO_ppitf_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_comb_flatGO_ppitf_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Combined FlatGO + PPITF + Dense
    echo  scrna-nn train --nn=flatGO_ppitf --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
    # Combined FlatGO + PPITF + Dense 100
    echo  scrna-nn train --nn=flatGO_ppitf 100 --fGO_ppitf_grps=data/flat_GO300_groups.txt,data/mouse_ppitf_groups.txt --with_dense=100  --siamese --out=${SCRATCH_PREFIX}${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense_100 --data=$TRAIN_DATA $EXTRA_OPTS >> ${SESSION_NAME}_commands.list
    echo ${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense_100 >> ${SESSION_NAME}_models.list
    echo ${DO_SCRATCH}cp -r ${SCRATCH_PREFIX}${MODEL_PREFIX}siam_comb_flatGO_ppitf_dense_100 ${MODEL_PREFIX} >> ${SESSION_NAME}_scratch_copy_commands.list
fi

NUM_JOBS=$(wc -l ${SESSION_NAME}_commands.list | awk {'print $1'})
echo "$NUM_JOBS"
NUM_JOBS=$(($NUM_JOBS - 1))
echo "$NUM_JOBS"
sbatch --array=0-$NUM_JOBS --mail-user ${EMAIL} --mail-type=END,FAIL slurm/train_model_arr.sh ${SESSION_NAME}_commands.list ${SESSION_NAME}_scratch_copy_commands.list
