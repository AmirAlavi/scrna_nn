from keras.models import Sequential
from keras.layers.core import Dense

def get_1layer_autoencoder(hidden_layer_size, input_dim):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=input_dim))
    model.add()
