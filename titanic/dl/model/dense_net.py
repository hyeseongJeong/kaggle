import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from numpy.random import seed

seed(0)
tf.random.set_seed(0)


def dense_net(input_shape=None, layer_dims=None, activation='relu', dropout_rate=0.2):
    input = Input(input_shape)
    x = input

    for layer_dim in layer_dims:
        x = Dense(layer_dim, activation=activation)(x)
        x = Dropout(rate=dropout_rate)(x)
        x = BatchNormalization()(x)

    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=y, name='dense_net')

    model.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model