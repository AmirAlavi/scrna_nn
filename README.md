# scrna-nn
A pipeline for reducing the dimensions of single-cell RNA-seq data using neural networks.

The pipeline exists as a script, installed via a python package.

`bin/scrna-nn` is the main script.
## Table of Contents
- [Installation](#installation)
- [Documentation (minimal)](#documentation)
- [Examples](#example-invocations)
- [Notes](#notes)
## Installation
You can install it into a python environment by cloning this repository, and from within its root folder, executing:
```
pip install .
```
After which, `scrna-nn` will be in your `PATH` when in your python environment, and can be invoked from anywhere.

## Documentation
`scrna-nn` has subcommands. `train`, `reduce`, and `retrieval` are the most heavily used, and most actively developed. Other subcommands may be added/removed in the future, and some are present but not working in their current state.

The script has help documentation:

```
$ scrna-nn -h
usage: scrna-nn [-h] {train,reduce,visualize,retrieval,analyze} ...

Single-cell RNA-seq dimensionality reduction using neural networks

optional arguments:
  -h, --help            show this help message and exit

subcommands:
  {train,reduce,visualize,retrieval,analyze}
    train               Train a scRNA-seq dimensionality reduction model.
    reduce              Use a trained model to reduce dimensions (embed)
                        scRNA-seq data.
    visualize           Visualize reduced dimension data.
    retrieval           Conduct a retrieval analysis experiment.
    analyze             Analyze data (incomplete).
```
There is more detailed help for each subcommand, and some subcommands have MANY options due to the experimental nature of this work.
```
$ scrna-nn reduce -h
usage: scrna-nn reduce [-h] [--data DATA] [--out OUT] [--save_meta]
                       trained_model_folder

positional arguments:
  trained_model_folder  Path to folder containing trained model.

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Path to input data. For 'train' command, this must be
                        a folder with train/valid/test files. (default: None)
  --out OUT             Path to save output to. For training and retrieval
                        this is a folder path. (default: None)
  --save_meta           Also save the metadata that was associated with the
                        input data with the reduced data (labels for the
                        samples, accession numbers for the samples). (default:
                        False)
```
## Example Invocations
First, we can train a basic Dense (aka Multi-Layer Perceptron) network. The network is trained to classify labeled cell types in the data. Later, this saved model can be used to embed data by using the last hidden layer as a learned embedding representation.
```
scrna-nn train --nn=dense 1000 100 --act=tanh --opt=sgd --epochs=100 --batch_size=256 --loss_history --checkpoints=val_loss --data=data_FOLDER --out=model_FOLDER
```
Then we can use the model to reduce some data:
```
scrna-nn reduce model_FOLDER --data=data_FILE.hdf5 --out=reduced_data_FILE.hdf5
```
Finally, we might want to do some retrieval testing in these reduced dimensions:
```
scrna-nn retrieval reduced_query_data_FILE.hdf5 reduced_database_data_FILE.hdf5 --out=retrieval_test_result_FOLDER
```
## Notes
- In the above examples, note that some arguments are expected to be FILEs vs FOLDERs. In particular, when using the `train` subcommand, the script expects the argument for the `--data` flag to be a folder, which should contain these three files:
  - `train_data.h5`
  - `valid_data.h5`
  - `test_data.h5`
