#!/bin/bash

module load cuda-8.0

CUDNN_ROOT=$HOME/cudnn/cuda
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDNN_ROOT/lib64:$LIBRARY_PATH

COMMANDS_FILE=$1

mapfile -t job_commands < $COMMANDS_FILE

eval ${job_commands[$SLURM_ARRAY_TASK_ID]}

