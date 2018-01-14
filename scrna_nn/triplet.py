# import pdb; pdb.set_trace()
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import SGD
from keras import backend as K
from theano import tensor as T

from .data_container import DataContainer


def base_net(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(1000, activation='tanh')(inputs)
    x = Dense(100, activation='tanh')(x)
    x = Dense(output_dim, activation='softmax')(x)
    base_net = Model(name="base_net", inputs=inputs, outputs=x)
    return base_net


def triplet_batch_hard_loss(y_true, y_pred):
    # y_pred is the embedding, y_true is the IDs (labels) of the samples (not 1-hot encoded)
    # They are mini-batched. If batch_size is B, and embedding dimension is D, shapes are:
    #   y_true: (B,)
    #   y_pred: (B,D)
    margin = 0.2
    
    # Get all-pairs distances
    y_true = K.sum(y_true, axis=1)
    diffs = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
    dist_mat = K.sqrt(K.sum(K.square(diffs), axis=-1) + K.epsilon())
    same_identity_mask = K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0))
    # TODO: make this backend-agnostic somehow
    negative_mask = T.bitwise_not(same_identity_mask)
    # XOR ensures that the same sample is paired with itself
    positive_mask = T.bitwise_xor(same_identity_mask, K.eye(256, dtype='bool'))
    #print(K.int_shape(y_true))
    #print(K.int_shape(y_pred))

    #positive_mask = T.bitwise_xor(same_identity_mask, T.eye(K.int_shape(y_true)[0]))

    furthest_positive = K.max(dist_mat*positive_mask, axis=1)
    #closest_negative = K.min(dist_mat*negative_mask + np.inf*same_identity_mask, axis=1)
    closest_negative = K.min(dist_mat*negative_mask + 1e6*same_identity_mask, axis=1)

    loss = K.maximum(furthest_positive - closest_negative + margin, 0)
    return loss
    

def get_triplet_net(base_net):
    embedding = base_net.layers[-2].output
    return Model(name="triplet_net", inputs=base_net.layers[0].input, outputs=embedding)
    

def main():
    # get a dataset
    dataset_file = '/home/aalavi/scrna_nn/data/mouse_data_20171220-154552/our_traindb_data.h5'
    data = DataContainer(dataset_file, False, True)
    X, y, _ = data.get_data()
    print("All data shape: ", X.shape)
    X = X[0:21760]
    y = y[0:21760]

    print(X.shape)
    print(y.shape)
    input_dim = X.shape[1]
    output_dim = max(y) + 1
    new_y = np.zeros((y.shape[0], 100), dtype=X.dtype)
    new_y[:,0] = y
    y = new_y
    base_model = base_net(input_dim, output_dim)
    triplet_model = get_triplet_net(base_model)
    print(triplet_model.summary())
    sgd = SGD(lr=0.1, nesterov=True)
    triplet_model.compile(loss=triplet_batch_hard_loss, optimizer=sgd)
    print("Successfully compiled model")
    X_train = X[0:20480]
    y_train = y[0:20480]
    X_valid = X[20480:21760]
    y_valid = y[20480:21760]
    history = triplet_model.fit(X_train, y_train, batch_size=256, epochs=25, verbose=1, validation_data=(X_valid, y_valid))
    

if __name__ == "__main__":
    main()
