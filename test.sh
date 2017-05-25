#!/bin/bash
#python scrna.py train 1layer_ae 100 --sn --gs
KERAS_BACKEND=theano python scrna.py train 2layer_ppitf 100 --sn --gs
echo "done"
