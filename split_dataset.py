"""Split Dataset

Usage:
    split_dataset.py <input_csv> [--train_fraction=<frac> --random]
    split_dataset.py (-h | --help)

Options:
    -h --help                Show this screen.
    --train_fraction=<frac>  Portion of the data to use for training split. Testing
                             split will be 1-<frac>. [default: 0.2]
    --random                 Whether to select samples randomly for each split.
"""
import math

from docopt import docopt
import numpy as np

from data_container import DataContainer

if __name__ == '__main__':
    args = docopt(__doc__)
    train_frac = float(args['--train_fraction'])
    # Load the original, whole dataset
    data = DataContainer(args['<input_csv>']) # no preprocessing
    # For supervised training, we only want the labeled data
    df = data.dataframe.loc[lambda df: df.Label != 'None', :]
    # Empty dataframes
    train_df = df.drop(df.index)
    test_df = df.drop(df.index)
    # Go through samples by dataset
    datasets = df.loc[:, 'Dataset'].values.astype(int)
    datasets = np.unique(datasets)
    for ds in datasets:
        print("On dataset: ", ds)
        # Get all samples from a dataset
        ds_df = df.loc[lambda df: df.Dataset == ds, :]
        num_samples = ds_df.shape[0]
        print("Has ", num_samples, " samples")
        # Select the training portion from this, and save the rest for test
        # (this way, both train and test have the same distribution of datasets)
        train = ds_df.sample(frac=train_frac)
        test = ds_df.drop(train.index)
        train_df = train_df.append(train, ignore_index=True)
        test_df = test_df.append(test, ignore_index=True)
    # Save the new datasets
    print("Writing out train data")
    train_df.to_csv("train.csv", sep='\t', index_label="Sample")
    print("Writing out test data")
    test_df.to_csv("test.csv", sep='\t', index_label="Sample")
    print("Done")
