import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


def create_model(class_num, input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
                                           include_top=True, weights='imagenet', input_tensor=None, pooling=None,
                                           classes=1000, classifier_activation=None)

    input = tf.keras.Input(input_shape)

    x = base(input)
    output = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)

    return model
