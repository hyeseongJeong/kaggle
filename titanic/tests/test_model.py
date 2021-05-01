from unittest import TestCase
from titanic.dl.model.dense_net import dense_net

import tensorflow as tf


class TestModel(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_dense_net(self):
        input_shape = (7,)
        layer_dims = (7, 14, 28)
        activation = 'relu'
        dropout_rate = 0.2

        model = dense_net(input_shape=input_shape, layer_dims=layer_dims, activation=activation, dropout_rate=dropout_rate)

        model.summary()

        batch_size = 10
        input = tf.random.normal(shape=(batch_size, input_shape[0]))
        output = model(input)

        self.assertEqual((10, 1), output.shape)
