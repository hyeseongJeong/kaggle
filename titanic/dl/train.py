import tensorflow as tf

from sklearn.model_selection import train_test_split
from titanic.data.load_dataset import load_titanic_csv_dataset
from titanic.dl.model.dense_net import dense_net


def main():
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    X_train, y_train, X_test = load_titanic_csv_dataset(dataset_dir, strategy='strategy_3')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
    model = dense_net(input_shape=(X_train.shape[1], ), layer_dims=(20,), activation='relu', dropout_rate=0.2)
    model.summary()

    tensorboard = tf.keras.callbacks.TensorBoard('./titanic/dl/logs', update_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=100000,
        min_delta=0.001,
        restore_best_weights=True,
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./titanic/dl/ckpt/',
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(x=X_train,
              y=y_train,
              verbose=0,
              epochs=100000,
              batch_size=16,
              callbacks=[early_stopping, tensorboard, model_checkpoint_callback],
              validation_data=(X_val, y_val))


if __name__ == '__main__':
    main()