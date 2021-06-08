
import tensorflow as tf
import numpy as np
import pathlib
import os


class DataSetHandler:
    def __init__(self, data_dir_path, img_height, img_width, batch_size, is_training=False):
        self.data_dir_path = pathlib.Path(data_dir_path)
        self.image_num = len(list(self.data_dir_path.glob('*.jpg')))
        self.class_names = np.array([item.name for item in self.data_dir_path.glob('*') if '.DS_Store' not in item.name])

        self.autotune = tf.data.experimental.AUTOTUNE
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.is_training = is_training

    def get_label(self, file_path):
        # convert the path to a list of path components
        name = tf.strings.split(file_path, os.path.sep)[-1]
        category = tf.strings.split(name, '.')[0]

        if category == 'cat':
            return 0
        elif category == 'dog':
            return 1
        else:
            return -1

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)

        # augmentation
        if self.is_training:
            img = tf.image.random_hue(img, 0.08)
            img = tf.image.random_saturation(img, 0.6, 1.6)
            img = tf.image.random_brightness(img, max_delta=0.08, seed=None)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_contrast(img, 0.7, 1.3)

        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize_with_pad(img,
                                        target_height=self.img_height,
                                        target_width=self.img_width)

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def get_processed_dataset(self, buffer_size=1000):
        list_ds = tf.data.Dataset.list_files(str(self.data_dir_path/'*'))

        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.autotune)
        labeled_ds = labeled_ds.shuffle(buffer_size=buffer_size)
        labeled_ds = labeled_ds.batch(self.batch_size)
        labeled_ds = labeled_ds.prefetch(buffer_size=self.autotune)

        return labeled_ds
