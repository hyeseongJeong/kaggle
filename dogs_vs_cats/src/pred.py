from absl import app, flags

import pandas as pd
import csv
import os
import cv2
import time
import shutil
import tensorflow as tf
import numpy as np


def load_img(path, img_height, img_width, img_ch):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=img_ch)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [img_height, img_width])


def main():
    df = None
    dl_model_path = '/Users/hyeseong/deep_learning/saved_models/dogs_vs_cats/best'
    data_path = '/Users/hyeseong/datasets/public/kaggle/dogs-vs-cats-mvml-2020/test/test'
    os.makedirs(os.path.join(data_path, 'not_sure'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'dog'), exist_ok=True)

    model = tf.keras.models.load_model(dl_model_path)
    _, img_h, img_w, img_ch = model.input_shape

    paths = sorted([os.path.join(data_path, name) for name in os.listdir(data_path) if '.jpg' in name])
    times = []

    for i, path in enumerate(paths):
        print(f'{i+1}/{len(paths)}')
        img = load_img(path, img_h, img_w, img_ch)
        st = time.time()
        pred = model.predict(img[tf.newaxis, ...])
        times.append(time.time()-st)

        prob = np.array(tf.reduce_max(pred), dtype=np.float)
        pred = np.array(tf.argmax(pred, axis=1), dtype=np.int)

        print(f'id: {int(os.path.splitext(os.path.basename(path))[0])}, label: {int(pred[0])}')
        _df = pd.DataFrame([{'id': int(os.path.splitext(os.path.basename(path))[0]),
                            'label': int(pred[0])}])
        if df is None:
            df = _df
        else:
            df = df.append(_df, ignore_index=True)
        df = df.sort_values(['id'], ascending=True)
        print(df)

        # if prob < 0.8:
        #     shutil.move(path, os.path.join(data_path, 'not_sure', os.path.basename(path)))
        # else:
        #     if pred == 0:
        #         shutil.move(path, os.path.join(data_path, 'cat', os.path.basename(path)))
        #     else:
        #         shutil.move(path, os.path.join(data_path, 'dog', os.path.basename(path)))
        print('\n')
    print('avg time: ', np.mean(times))
    df.to_csv('/tmp/result.csv', index=False, header=True)


if __name__ == '__main__':
    # main()
    data = pd.read_csv('/tmp/result.csv')
    print(data)