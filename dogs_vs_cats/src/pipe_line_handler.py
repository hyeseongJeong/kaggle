import os
import tensorflow as tf
import numpy as np
from distutils.dir_util import copy_tree
from collections import defaultdict
from datetime import datetime
from pytz import timezone
from dogs_vs_cats.src.model import create_model
from dogs_vs_cats.src.data import DataSetHandler


@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        pred = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, pred, from_logits=False)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(tf.cast(labels, dtype=tf.float32), pred)

    return loss


@tf.function
def valid_step(images, labels, model, valid_loss, valid_accuracy):
    pred = model(images)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, pred, from_logits=False)
    loss = tf.reduce_mean(loss)

    valid_loss(loss)
    valid_accuracy(labels, pred)

    return loss, pred


class PipeLineHandler:
    def __init__(self, trainee_info_dict):
        self.trainee_info_dict = trainee_info_dict
        self.dataset_path = self.trainee_info_dict['dataset_path']
        self.pretrained_model_path = self.trainee_info_dict['pretrained_model_path']
        self.saved_model_path = self.trainee_info_dict['saved_model_path']
        self.img_height = self.trainee_info_dict['img_height']
        self.img_width = self.trainee_info_dict['img_width']
        self.batch_size = self.trainee_info_dict['batch_size']
        self.class_num = self.trainee_info_dict['class_num']
        self.epochs = self.trainee_info_dict['epochs']

    def run(self):
        # dataset setting
        train_ds_handler = DataSetHandler(os.path.join(self.dataset_path, 'train'), self.img_height, self.img_width, self.batch_size, is_training=True)
        valid_ds_handler = DataSetHandler(os.path.join(self.dataset_path, 'valid'), self.img_height, self.img_width, self.batch_size)
        train_ds = train_ds_handler.get_processed_dataset()
        valid_ds = valid_ds_handler.get_processed_dataset()
        train_image_num = train_ds_handler.image_num
        valid_image_num = valid_ds_handler.image_num

        # optimizer & learning rate setting
        boundaries = [10, 20]
        values = [0.0005, 0.0002, 0.0001]
        learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # model setting
        try:
            model = tf.keras.models.load_model(self.pretrained_model_path)
            print('**** pretrained model build ****')
        except Exception as e:
            model = create_model(class_num=self.class_num,
                                 input_shape=(self.img_height, self.img_width, 3))
            print('**** new model build ****')

        # metric setting
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

        # tensorboard setting
        # display graph
        train_log_dir = os.path.join(self.saved_model_path, f'events/{datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")}/train')
        valid_log_dir = os.path.join(self.saved_model_path, f'events/{datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")}/valid')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        # check point setting
        ckpt_path = os.path.join(self.saved_model_path, 'checkpoints', str(datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")))

        # training
        best_accuracy = 0
        for epoch in range(self.epochs):
            for step, (images, labels) in enumerate(train_ds):
                train_step(images, labels, model, optimizer, train_loss, train_accuracy)
                if step % 2 == 0:
                    print('train, epoch:{}/{} step:{}/{}'.format(epoch, self.epochs, step, int(train_image_num/self.batch_size)))
            print()

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            results_dict = defaultdict(list)
            for step, (valid_images, valid_labels) in enumerate(valid_ds):
                _, pred = valid_step(valid_images, valid_labels, model, valid_loss, valid_accuracy)
                for _l, _p, _img in zip(valid_labels, pred, valid_images):
                    results_dict[f'label:{int(_l)}|pred:{str(np.array(tf.math.argmax(_p), dtype=np.int))}'].append(_img)

                if step % 2 == 0:
                    print('valid, epoch:{}/{} step:{}/{}'.format(epoch, self.epochs, step, int(valid_image_num/self.batch_size)))
            print()

            with valid_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)
            with valid_summary_writer.as_default():
                for _k, _v in results_dict.items():
                    tf.summary.image(_k, _v, step=epoch, max_outputs=30)

            if valid_accuracy.result() > best_accuracy:
                best_accuracy = valid_accuracy.result()

                model.save(f'{ckpt_path}/best')
                copy_tree(f'{ckpt_path}/best', f'{ckpt_path}/best.backup')
