from sparse_layer_new import Sparse
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.utils import to_categorical
input_dim = 3
hidden_size = 4
adj = np.array([[0, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 1]])
print("Adjacency matrix:")
print(adj)
print()
orig = adj
blah = Sparse(adjacency_mat=adj)
inputs = Input(shape=(input_dim,))
x = blah(inputs)
output_dim = 5
x = Dense(output_dim, activation='softmax')(x)
model = Model(inputs=inputs, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print()
print(model.summary())
print()

# Sample inputs
num_samples = 100
X = np.random.rand(num_samples, input_dim)
y = np.random.randint(low=0, high=output_dim, size=num_samples)
y = to_categorical(y, num_classes=output_dim)
print("Sparse kernel before training:")
print(model.layers[1].get_weights()[0])
print(model.layers[1].get_weights()[2])
print()
model.fit(X, y, epochs = 5, batch_size=10)
print("Sparse kernel after training:")
print(model.layers[1].get_weights()[0])
print(model.layers[1].get_weights()[2])
print()
model.fit(X, y, epochs = 50, batch_size=10)
print("Sparse kernel after even more training:")
print(model.layers[1].get_weights()[0])
print(model.layers[1].get_weights()[2])
print()
model.save("testing_sparse_save.h5")
print("\nSAVING MODEL, DELETING IN-MEMORY MODEL\n")
del model
del adj
print("LOADING MODEL FROM DISK")
print()
model = load_model("testing_sparse_save.h5", custom_objects={'Sparse':Sparse})
print(model.summary())
print()
print("Sparse kernel after even more training:")
print(model.layers[1].get_weights()[0])
print(model.layers[1].get_weights()[2])
print()
print(model.layers[1].adjacency_mat)
print()
print(orig == model.layers[1].adjacency_mat)
