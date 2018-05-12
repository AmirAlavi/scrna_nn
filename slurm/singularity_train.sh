#!/bin/bash
source ~/singularity_locale_exports.sh
source tfl_venv/bin/activate
scrna-nn train --nn=dense 1136 500 100 --sn --siamese --out=models/tflow_large --data=data/mouse_data_20170914-193533_42682_cells/traindb_data.h5 --epochs=5 --batch_size=2048 --ngpus=1
