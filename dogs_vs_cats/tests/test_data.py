from unittest import TestCase
from dogs_vs_cats.data import DataSetHandler
import os
import tensorflow as tf


class TestData(TestCase):
    def setUp(self) -> None:
        self.data_dir_path = '/Users/hyeseong/datasets/public/kaggle/dogs-vs-cats/train'
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 30
        self.is_training = False
        self.paths = sorted([os.path.join(self.data_dir_path, name) for name in os.listdir(self.data_dir_path) if '.jpg' in name])

        self.handler = DataSetHandler(self.data_dir_path,
                                      self.img_height,
                                      self.img_width,
                                      self.batch_size,
                                      self.is_training)

    def test_get_label(self):
        label = self.handler.get_label(self.paths[100])
        print(label)
        self.assertEqual(int(self.paths[0].split('.')[-2]), int(label))

    def test_decode_img(self):
        img = tf.io.read_file(self.paths[0])
        img = self.handler.decode_img(img)

        print(img.shape)

    def test_get_processed_dataset(self):
        label_ds = self.handler.get_processed_dataset()
        for image, label in label_ds:
            print(label)