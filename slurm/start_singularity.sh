#!/bin/bash

# Basic help message:
if [ $# -eq 0 ]; then
    printf "Usage:\n start_singularity.sh <image> <virtualenv>\n"
    exit 1
fi

IMAGE=$1
VENV=$2

source singularity shell ${IMAGE}
source ${VENV}/bin/activate
echo "Singularity image ${IMAGE} started"
echo "Activated virtual environment ${VENV}"
