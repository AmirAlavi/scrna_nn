import keras
from keras.models import Model
from keras.layers import Input

from .sparse_autoencoder import DenseLayerAutoencoder, SparseLayerAutoencoder

def get_ae_model(model_name, latent_layer_size, input_dim, activation_fcn='tanh', adj_mat=None):
    inputs = Input(shape=(input_dim,))
    x = inputs
    if model_name == 'dense':
        x = DenseLayerAutoencoder(latent_layer_size, activation=activation_fcn)(x)
    else:
        # assuming it's some sort of sparsely connected architecture otherwise
        x = SparseLayerAutoencoder(activation=activation_fcn, adjacency_mat=adj_mat)(x)
    model = Model(inputs=inputs, outputs=x)
    return model
