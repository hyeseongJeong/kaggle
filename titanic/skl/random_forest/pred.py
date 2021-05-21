from titanic.data.load_dataset import load_titanic_csv_dataset

import joblib
import pandas as pd
import os


def main():
    model = joblib.load('./model.pkl')

    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    X_train, y_train, X_test = load_titanic_csv_dataset(dataset_dir, strategy='strategy_3')

    y_pred = model.predict(X_test)

    test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    submission = pd.DataFrame({
        'PassengerId': test_df["PassengerId"],
        'Survived': y_pred.flatten()
    })
    print(submission)
    submission.to_csv('./submission.csv', index=False)


if __name__ == "__main__":
    main()

