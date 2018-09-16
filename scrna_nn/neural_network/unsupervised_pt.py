from os.path import join

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Model

from sparsely_connected_keras import Sparse
from tied_autoencoder_keras import DenseLayerAutoencoder, SparseLayerAutoencoder

NOISE_LEVEL = 0.1

def get_embedder(model, i, opt, args):
    #inputs = Input(shape=(input_dim,))
    embedder = Model(inputs=model.layers[0].input, outputs=model.layers[i-1].output)
    print("Embedder architecture:")
    print(embedder.summary())
    embedder.compile(loss='mean_squared_error', optimizer=opt)
    return embedder

def pretrain_model(model, input_dim, opt, X_orig, working_dir, args):
    for i in range(1, len(model.layers)):
        print("GLUP layer {}".format(i))
        if not isinstance(model.layers[i], (Dense, Sparse)):
            print("layer {}:{} is not a Dense or Sparse layer, skipping".format(i, type(model.layers[i])))
            continue
        embedder = get_embedder(model, i, opt, args)
        embedded_data = embedder.predict(X_orig)
        corrupted = embedded_data + NOISE_LEVEL * \
        np.random.normal(loc=0, scale=1, size=embedded_data.shape)
        inputs = Input(shape=(embedded_data.shape[1],))
        if isinstance(model.layers[i], Sparse):
            x = SparseLayerAutoencoder(
                activation=args.act,
                adjacency_mat=model.layers[i].adjacency_mat)(inputs)
        else:
            x = DenseLayerAutoencoder([model.layers[i].output_shape[-1]], activation=args.act)(inputs)
        dae = Model(inputs=inputs, outputs=x)
        print("DAE architecture:")
        print(dae.summary())
        dae.compile(loss='mean_squared_error', optimizer=opt)
        callbacks_list = []
        if args.early_stop:
            print("using early stopping")
            callbacks_list = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    verbose=1)]
        dae.fit(
            embedded_data,
            corrupted,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_split=args.valid,
            callbacks=callbacks_list)
        if isinstance(model.layers[i], Sparse):
            model.layers[i].set_weights(dae.layers[1].get_weights()[1:])
        else:
            model.layers[i].set_weights(dae.layers[1].get_weights()[:2])
        model.layers[i].trainable = False
    model.save_weights(join(working_dir, 'pretrained_layer_weights.h5'))
