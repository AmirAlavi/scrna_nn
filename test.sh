#!/bin/bash
#python scrna.py train 1layer_ae --sn --gs 100
KERAS_BACKEND=theano python scrna.py train 2layer_ppitf --sn --gs 100
echo "done"
