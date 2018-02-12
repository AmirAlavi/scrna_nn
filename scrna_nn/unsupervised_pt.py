import pickle
from os.path import join
import sys

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping

from .sparse_layer import Sparse
from .sparse_autoencoder import DenseLayerAutoencoder, SparseLayerAutoencoder
from .util import ScrnaException

NOISE_LEVEL = 0.1

def pretrain_dense_1136_100_model(input_dim, opt, X_orig, working_dir, args):
    print("Pretraining Dense 1136 100 architecture")
    callbacks_list = []
    if args['--early_stop']:
        print("using early stopping")
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
    # h1 pretraining
    inputs = Input(shape=(input_dim,))
    x = DenseLayerAutoencoder(1136, activation=args['--act'])(inputs)
    h1 = Model(inputs=inputs, outputs=x)

    # compile and train h1
    h1.compile(loss='mean_squared_error', optimizer=opt)
    X = X_orig + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig.shape)
    y = X_orig
    h1.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)


    # 2nd hidden layer
    # First get new clean data
    inputs = Input(shape=(input_dim,))
    x = Dense(1136, activation=args['--act'])(inputs)
    h2_rep = Model(inputs=inputs, outputs=x)
    h2_rep.layers[1].set_weights(h1.layers[1].get_weights()[1:])
    h2_rep.layers[1].trainable = False
    h2_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig2 = h2_rep.predict(X_orig)
    X = X_orig2 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig2.shape)
    y = X_orig2

    inputs = Input(shape=(1136,))
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    h2 = Model(inputs=inputs, outputs=x)

    # compile and train h2
    h2.compile(loss='mean_squared_error', optimizer=opt)
    h2.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # finally, write out the pretrained weights
    pretrained_layers = [h1.layers[1].get_weights()[1:], h2.layers[1].get_weights()[1:]]
    with open(join(working_dir, "pretrained_layers.p"), 'wb') as f:
        pickle.dump(pretrained_layers, f)
    sys.exit()

def pretrain_dense_1136_500_100_model(input_dim, opt, X_orig, working_dir, args):
    print("Pretraining Dense 1136 500 100 architecture")
    callbacks_list = []
    if args['--early_stop']:
        print("using early stopping")
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
    # h1 pretraining
    inputs = Input(shape=(input_dim,))
    x = DenseLayerAutoencoder(1136, activation=args['--act'])(inputs)
    h1 = Model(inputs=inputs, outputs=x)

    # compile and train h1
    h1.compile(loss='mean_squared_error', optimizer=opt)
    X = X_orig + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig.shape)
    y = X_orig
    h1.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)


    # 2nd hidden layer
    # First get new clean data
    inputs = Input(shape=(input_dim,))
    x = Dense(1136, activation=args['--act'])(inputs)
    h2_rep = Model(inputs=inputs, outputs=x)
    h2_rep.layers[1].set_weights(h1.layers[1].get_weights()[1:])
    h2_rep.layers[1].trainable = False
    h2_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig2 = h2_rep.predict(X_orig)
    X = X_orig2 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig2.shape)
    y = X_orig2

    inputs = Input(shape=(1136,))
    x = DenseLayerAutoencoder(500, activation=args['--act'])(inputs)
    h2 = Model(inputs=inputs, outputs=x)

    # compile and train h2
    h2.compile(loss='mean_squared_error', optimizer=opt)
    h2.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # 3rd hidden layer
    # First get new clean data
    inputs = Input(shape=(input_dim,))
    x = Dense(1136, activation=args['--act'])(inputs)
    x = Dense(500, activation=args['--act'])(x)
    h3_rep = Model(inputs=inputs, outputs=x)
    h3_rep.layers[1].set_weights(h1.layers[1].get_weights()[1:])
    h3_rep.layers[1].trainable = False
    h3_rep.layers[2].set_weights(h2.layers[1].get_weights()[1:])
    h3_rep.layers[2].trainable = False
    h3_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig3 = h3_rep.predict(X_orig)
    X = X_orig3 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig3.shape)
    y = X_orig3

    inputs = Input(shape=(500,))
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    h3 = Model(inputs=inputs, outputs=x)

    # compile and train h3
    h3.compile(loss='mean_squared_error', optimizer=opt)
    h3.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # finally, write out the pretrained weights
    pretrained_layers = [h1.layers[1].get_weights()[1:], h2.layers[1].get_weights()[1:], h3.layers[1].get_weights()[1:]]
    with open(join(working_dir, "pretrained_layers.p"), 'wb') as f:
        pickle.dump(pretrained_layers, f)
    sys.exit()


def pretrain_ppitf_1136_100_model(input_dim, adj_mat, opt, X_orig, working_dir, args):
    print("Pretraining PPITF 1136 100 architecture")        
    callbacks_list = []
    if args['--early_stop']:
        print("using early stopping")
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
    # sparse units pretraining
    inputs = Input(shape=(input_dim,))
    x = SparseLayerAutoencoder(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    sparse = Model(inputs=inputs, outputs=x)

    # compile and train sparse
    sparse.compile(loss='mean_squared_error', optimizer=opt)
    X = X_orig + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig.shape)
    y = X_orig
    sparse.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # dense units pretraining
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    dense = Model(inputs=inputs, outputs=x)

    # compile and train dense
    dense.compile(loss='mean_squared_error', optimizer=opt)
    dense.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # 2nd hidden layer
    inputs = Input(shape=(input_dim,))
    sparse_out = Sparse(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    dense_out = Dense(100, activation=args['--act'])(inputs)
    x = keras.layers.concatenate([sparse_out, dense_out])
    h2_rep = Model(inputs=inputs, outputs=x)
    h2_rep.layers[1].set_weights(sparse.layers[1].get_weights()[1:])
    h2_rep.layers[1].trainable = False
    h2_rep.layers[2].set_weights(dense.layers[1].get_weights()[1:])
    h2_rep.layers[2].trainable = False
    h2_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig2 = h2_rep.predict(X_orig)
    X = X_orig2 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig2.shape)
    y = X_orig2

    inputs = Input(shape=(1136,))
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    h2 = Model(inputs=inputs, outputs=x)

    # compile and train h2
    h2.compile(loss='mean_squared_error', optimizer=opt)
    h2.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # finally, write out the pretrained weights
    pretrained_layers = [sparse.layers[1].get_weights()[1:], dense.layers[1].get_weights()[1:], h2.layers[1].get_weights()[1:]]
    with open(join(working_dir, "pretrained_layers.p"), 'wb') as f:
        pickle.dump(pretrained_layers, f)
    sys.exit()

def pretrain_ppitf_1136_500_100_model(input_dim, adj_mat, opt, X_orig, working_dir, args):
    print("Pretraining PPITF 1136 500 100 architecture")
    callbacks_list = []
    if args['--early_stop']:
        print("using early stopping")
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
    # sparse units pretraining
    inputs = Input(shape=(input_dim,))
    x = SparseLayerAutoencoder(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    sparse = Model(inputs=inputs, outputs=x)

    # compile and train sparse
    sparse.compile(loss='mean_squared_error', optimizer=opt)
    X = X_orig + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig.shape)
    y = X_orig
    sparse.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # dense units pretraining
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    dense = Model(inputs=inputs, outputs=x)

    # compile and train dense
    dense.compile(loss='mean_squared_error', optimizer=opt)
    dense.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # 2nd hidden layer
    inputs = Input(shape=(input_dim,))
    sparse_out = Sparse(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    dense_out = Dense(100, activation=args['--act'])(inputs)
    x = keras.layers.concatenate([sparse_out, dense_out])
    h2_rep = Model(inputs=inputs, outputs=x)
    h2_rep.layers[1].set_weights(sparse.layers[1].get_weights()[1:])
    h2_rep.layers[1].trainable = False
    h2_rep.layers[2].set_weights(dense.layers[1].get_weights()[1:])
    h2_rep.layers[2].trainable = False
    h2_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig2 = h2_rep.predict(X_orig)
    X = X_orig2 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig2.shape)
    y = X_orig2

    inputs = Input(shape=(1136,))
    x = DenseLayerAutoencoder(500, activation=args['--act'])(inputs)
    h2 = Model(inputs=inputs, outputs=x)

    # compile and train h2
    h2.compile(loss='mean_squared_error', optimizer=opt)
    h2.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # 3rd hidden layer
    inputs = Input(shape=(input_dim,))
    sparse_out = Sparse(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    dense_out = Dense(100, activation=args['--act'])(inputs)
    x = keras.layers.concatenate([sparse_out, dense_out])
    x = Dense(500, activation=args['--act'])(x)
    h3_rep = Model(inputs=inputs, outputs=x)
    h3_rep.layers[1].set_weights(sparse.layers[1].get_weights()[1:])
    h3_rep.layers[1].trainable = False
    h3_rep.layers[2].set_weights(dense.layers[1].get_weights()[1:])
    h3_rep.layers[2].trainable = False
    h3_rep.layers[4].set_weights(h2.layers[1].get_weights()[1:])
    h3_rep.layers[4].trainable = False
    h3_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig3 = h3_rep.predict(X_orig)
    X = X_orig3 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig3.shape)
    y = X_orig3

    inputs = Input(shape=(500,))
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    h3 = Model(inputs=inputs, outputs=x)
    

    # compile and train h3
    h3.compile(loss='mean_squared_error', optimizer=opt)
    h3.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # finally, write out the pretrained weights
    pretrained_layers = [sparse.layers[1].get_weights()[1:], dense.layers[1].get_weights()[1:], h2.layers[1].get_weights()[1:], h3.layers[1].get_weights()[1:]]
    with open(join(working_dir, "pretrained_layers.p"), 'wb') as f:
        pickle.dump(pretrained_layers, f)
    sys.exit()

def pretrain_flatGO_400_100_model(input_dim, adj_mat, opt, X_orig, working_dir, args):
    print("Pretraining FlatGO 400 100 architecture")        
    callbacks_list = []
    if args['--early_stop']:
        print("using early stopping")
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
    # sparse units pretraining
    inputs = Input(shape=(input_dim,))
    x = SparseLayerAutoencoder(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    sparse = Model(inputs=inputs, outputs=x)

    # compile and train sparse
    sparse.compile(loss='mean_squared_error', optimizer=opt)
    X = X_orig + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig.shape)
    y = X_orig
    sparse.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # dense units pretraining
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    dense = Model(inputs=inputs, outputs=x)

    # compile and train dense
    dense.compile(loss='mean_squared_error', optimizer=opt)
    dense.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # 2nd hidden layer
    inputs = Input(shape=(input_dim,))
    sparse_out = Sparse(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    dense_out = Dense(100, activation=args['--act'])(inputs)
    x = keras.layers.concatenate([sparse_out, dense_out])
    h2_rep = Model(inputs=inputs, outputs=x)
    h2_rep.layers[1].set_weights(sparse.layers[1].get_weights()[1:])
    h2_rep.layers[1].trainable = False
    h2_rep.layers[2].set_weights(dense.layers[1].get_weights()[1:])
    h2_rep.layers[2].trainable = False
    h2_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig2 = h2_rep.predict(X_orig)
    X = X_orig2 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig2.shape)
    y = X_orig2

    inputs = Input(shape=(400,))
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    h2 = Model(inputs=inputs, outputs=x)

    # compile and train h2
    h2.compile(loss='mean_squared_error', optimizer=opt)
    h2.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # finally, write out the pretrained weights
    pretrained_layers = [sparse.layers[1].get_weights()[1:], dense.layers[1].get_weights()[1:], h2.layers[1].get_weights()[1:]]
    with open(join(working_dir, "pretrained_layers.p"), 'wb') as f:
        pickle.dump(pretrained_layers, f)
    sys.exit()

def pretrain_flatGO_400_200_100_model(input_dim, adj_mat, opt, X_orig, working_dir, args):
    print("Pretraining FlatGO 400 200 100 architecture")
    callbacks_list = []
    if args['--early_stop']:
        print("using early stopping")
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
    # sparse units pretraining
    inputs = Input(shape=(input_dim,))
    x = SparseLayerAutoencoder(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    sparse = Model(inputs=inputs, outputs=x)

    # compile and train sparse
    sparse.compile(loss='mean_squared_error', optimizer=opt)
    X = X_orig + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig.shape)
    y = X_orig
    sparse.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # dense units pretraining
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    dense = Model(inputs=inputs, outputs=x)

    # compile and train dense
    dense.compile(loss='mean_squared_error', optimizer=opt)
    dense.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # 2nd hidden layer
    inputs = Input(shape=(input_dim,))
    sparse_out = Sparse(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    dense_out = Dense(100, activation=args['--act'])(inputs)
    x = keras.layers.concatenate([sparse_out, dense_out])
    h2_rep = Model(inputs=inputs, outputs=x)
    h2_rep.layers[1].set_weights(sparse.layers[1].get_weights()[1:])
    h2_rep.layers[1].trainable = False
    h2_rep.layers[2].set_weights(dense.layers[1].get_weights()[1:])
    h2_rep.layers[2].trainable = False
    h2_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig2 = h2_rep.predict(X_orig)
    X = X_orig2 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig2.shape)
    y = X_orig2

    inputs = Input(shape=(400,))
    x = DenseLayerAutoencoder(200, activation=args['--act'])(inputs)
    h2 = Model(inputs=inputs, outputs=x)

    # compile and train h2
    h2.compile(loss='mean_squared_error', optimizer=opt)
    h2.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # 3rd hidden layer
    inputs = Input(shape=(input_dim,))
    sparse_out = Sparse(activation=args['--act'], adjacency_mat=adj_mat)(inputs)
    dense_out = Dense(100, activation=args['--act'])(inputs)
    x = keras.layers.concatenate([sparse_out, dense_out])
    x = Dense(200, activation=args['--act'])(x)
    h3_rep = Model(inputs=inputs, outputs=x)
    h3_rep.layers[1].set_weights(sparse.layers[1].get_weights()[1:])
    h3_rep.layers[1].trainable = False
    h3_rep.layers[2].set_weights(dense.layers[1].get_weights()[1:])
    h3_rep.layers[2].trainable = False
    h3_rep.layers[4].set_weights(h2.layers[1].get_weights()[1:])
    h3_rep.layers[4].trainable = False
    h3_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig3 = h3_rep.predict(X_orig)
    X = X_orig3 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig3.shape)
    y = X_orig3

    inputs = Input(shape=(200,))
    x = DenseLayerAutoencoder(100, activation=args['--act'])(inputs)
    h3 = Model(inputs=inputs, outputs=x)
    

    # compile and train h3
    h3.compile(loss='mean_squared_error', optimizer=opt)
    h3.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # finally, write out the pretrained weights
    pretrained_layers = [sparse.layers[1].get_weights()[1:], dense.layers[1].get_weights()[1:], h2.layers[1].get_weights()[1:], h3.layers[1].get_weights()[1:]]
    with open(join(working_dir, "pretrained_layers.p"), 'wb') as f:
        pickle.dump(pretrained_layers, f)
    sys.exit()


def pretrain_GOlvls_model(input_dim, level1_adj_mat, level2_adj_mat, level3_adj_mat, opt, X_orig, working_dir, args):
    # GO dimensions:
    # 20499
    #  5394
    #   925
    #    69
    print("Pretraining GO architecture")
    callbacks_list = []
    if args['--early_stop']:
        print("using early stopping")
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
    # 1st level pretraining
    inputs = Input(shape=(input_dim,))
    x = SparseLayerAutoencoder(activation=args['--act'], adjacency_mat=level1_adj_mat)(inputs)
    level1 = Model(inputs=inputs, outputs=x)

    # compile and train level 1
    level1.compile(loss='mean_squared_error', optimizer=opt)
    X = X_orig + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig.shape)
    y = X_orig
    level1.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)

    # 2nd level pretraining
    inputs = Input(shape=(input_dim,))
    x = Sparse(activation=args['--act'], adjacency_mat=level1_adj_mat)(inputs)
    level2_rep = Model(inputs=inputs, outputs=x)
    level2_rep.layers[1].set_weights(level1.layers[1].get_weights()[1:])
    level2_rep.layers[1].trainable = False
    level2_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig2 = level2_rep.predict(X_orig)
    X = X_orig2 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig2.shape)
    y = X_orig2

    inputs = Input(shape=(5394,))
    x = SparseLayerAutoencoder(activation=args['--act'], adjacency_mat=level2_adj_mat)(inputs)
    level2 = Model(inputs=inputs, outputs=x)
    
    # compile and train level 2
    level2.compile(loss='mean_squared_error', optimizer=opt)
    level2.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)
    
    # 3rd level pretraining
    inputs = Input(shape=(input_dim,))
    x = Sparse(activation=args['--act'], adjacency_mat=level1_adj_mat)(inputs)
    x = Sparse(activation=args['--act'], adjacency_mat=level2_adj_mat)(x)
    level3_rep = Model(inputs=inputs, outputs=x)
    level3_rep.layers[1].set_weights(level1.layers[1].get_weights()[1:])
    level3_rep.layers[1].trainable = False
    level3_rep.layers[2].set_weights(level2.layers[1].get_weights()[1:])
    level3_rep.layers[2].trainable = False
    level3_rep.compile(loss='mean_squared_error', optimizer=opt)
    X_orig3 = level3_rep.predict(X_orig)
    X = X_orig3 + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig3.shape)
    y = X_orig3

    inputs = Input(shape=(925,))
    x = SparseLayerAutoencoder(activation=args['--act'], adjacency_mat=level3_adj_mat)(inputs)
    level3 = Model(inputs=inputs, outputs=x)

    # compile and train level 3
    level3.compile(loss='mean_squared_error', optimizer=opt)
    level3.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)
    
    # dense pretraining
    inputs = Input(shape=(input_dim,))
    x = DenseLayerAutoencoder(31, activation=args['--act'])(inputs)
    dense = Model(inputs=inputs, outputs=x)

    # compile and train dense
    dense.compile(loss='mean_squared_error', optimizer=opt)
    X = X_orig + NOISE_LEVEL * np.random.normal(loc=0, scale=1, size=X_orig.shape)
    y = X_orig
    dense.fit(X, y, batch_size=int(args['--batch_size']), epochs=int(args['--epochs']), verbose=1, validation_split=float(args['--valid']), callbacks=callbacks_list)
    
    # finally, write out the pretrained weights
    pretrained_layers = [level1.layers[1].get_weights()[1:], level2.layers[1].get_weights()[1:], level3.layers[1].get_weights()[1:], dense.layers[1].get_weights()[1:]]
    with open(join(working_dir, "pretrained_layers.p"), 'wb') as f:
        pickle.dump(pretrained_layers, f)
    sys.exit()


def set_dense_1136_100_weights(model, pretrained_folder):
    with open(join(pretrained_folder, "pretrained_layers.p"), 'rb') as f:
        pt_weights = pickle.load(f)
    model.layers[1].set_weights(pt_weights[0])
    model.layers[2].set_weights(pt_weights[1])

def set_dense_1136_500_100_weights(model, pretrained_folder):
    with open(join(pretrained_folder, "pretrained_layers.p"), 'rb') as f:
        pt_weights = pickle.load(f)
    model.layers[1].set_weights(pt_weights[0])
    model.layers[2].set_weights(pt_weights[1])
    model.layers[3].set_weights(pt_weights[2])

def set_ppitf_1136_100_weights(model, pretrained_folder):
    with open(join(pretrained_folder, "pretrained_layers.p"), 'rb') as f:
        pt_weights = pickle.load(f)
    model.layers[1].set_weights(pt_weights[0])
    model.layers[2].set_weights(pt_weights[1])
    model.layers[4].set_weights(pt_weights[2])

def set_ppitf_1136_500_100_weights(model, pretrained_folder):
    with open(join(pretrained_folder, "pretrained_layers.p"), 'rb') as f:
        pt_weights = pickle.load(f)
    model.layers[1].set_weights(pt_weights[0])
    model.layers[2].set_weights(pt_weights[1])
    model.layers[4].set_weights(pt_weights[2])
    model.layers[5].set_weights(pt_weights[3])

def set_flatGO_400_100_weights(model, pretrained_folder):
    with open(join(pretrained_folder, "pretrained_layers.p"), 'rb') as f:
        pt_weights = pickle.load(f)
    model.layers[1].set_weights(pt_weights[0])
    model.layers[2].set_weights(pt_weights[1])
    model.layers[4].set_weights(pt_weights[2])

def set_flatGO_400_200_100_weights(model, pretrained_folder):
    with open(join(pretrained_folder, "pretrained_layers.p"), 'rb') as f:
        pt_weights = pickle.load(f)
    model.layers[1].set_weights(pt_weights[0])
    model.layers[2].set_weights(pt_weights[1])
    model.layers[4].set_weights(pt_weights[2])
    model.layers[5].set_weights(pt_weights[3])

def set_GOlvls_weights(model, pretrained_folder):
    with open(join(pretrained_folder, "pretrained_layers.p"), 'rb') as f:
        pt_weights = pickle.load(f)
    model.layers[1].set_weights(pt_weights[0])
    model.layers[2].set_weights(pt_weights[1])
    model.layers[3].set_weights(pt_weights[2])
    model.layers[4].set_weights(pt_weights[3])
    
def set_pretrained_weights(model, args, pretrained_folder):
    print("Setting pretrained weights from unsupervised layer-wise pretraining of: ", pretrained_folder)
    hidden_layer_sizes = [int(x) for x in args['<hidden_layer_sizes>']]
    if args['--nn'] == 'dense':
        if hidden_layer_sizes == [1136, 100]:
            set_dense_1136_100_weights(model, pretrained_folder)
        elif hidden_layer_sizes == [1136, 500, 100]:
            set_dense_1136_500_100_weights(model, pretrained_folder)
        else:
            raise ScrnaException("Layerwise pretraining not implemented for this architecture")
    elif args['--nn'] == 'sparse' and int(args['--with_dense']) == 100:
        if 'flat' in args['--sparse_groupings']:
            print('Using pretrained weights for FlatGO')
            if hidden_layer_sizes == [100]:
                set_flatGO_400_100_weights(model, pretrained_folder)
            elif hidden_layer_sizes == [200, 100]:
                set_flatGO_400_200_100_weights(model, pretrained_folder)
            else:
                raise ScrnaException("Layerwise pretraining not implemented for this architecture")
        else:
            print("Using pretrained weights for PPITF")
            if hidden_layer_sizes == [100]:
                set_ppitf_1136_100_weights(model, pretrained_folder)
            elif hidden_layer_sizes == [500, 100]:
                set_ppitf_1136_500_100_weights(model, pretrained_folder)
            else:
                raise ScrnaException("Layerwise pretraining not implemented for this architecture")
    elif args['--nn'] == 'GO' and int(args['--with_dense']) == 31:
        set_GOlvls_weights(model, pretrained_folder)
    else:
        raise ScrnaException("Layerwise pretraining not implemented for this architecture")
