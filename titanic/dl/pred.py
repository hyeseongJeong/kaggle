from titanic.data.load_dataset import load_titanic_csv_dataset

import tensorflow as tf
import pandas as pd
import os


def main():
    model = tf.keras.models.load_model('./dl/ckpt/')
    model.summary()

    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    X_train, y_train, X_test = load_titanic_csv_dataset(dataset_dir, strategy='strategy_1')

    y_prob = model.predict(x=X_test)
    y_pred = (y_prob > 0.5) * 1

    test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    submission = pd.DataFrame({
        'PassengerId': test_df["PassengerId"],
        'Survived': y_pred.flatten()
    })
    print(submission)
    submission.to_csv('./dl/submission.csv', index=False)


if __name__ == "__main__":
    main()

