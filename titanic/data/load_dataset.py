from titanic.data.preprocess import strategy_1, strategy_2, strategy_3
import os
import pandas as pd


def load_titanic_csv_dataset(directory, strategy='strategy_1'):
    try:
        if strategy == 'strategy_1':
            load_fn = strategy_1
        elif strategy == 'strategy_2':
            load_fn = strategy_2
        elif strategy == 'strategy_3':
            load_fn = strategy_3
        else:
            load_fn = None

        train_and_test_data = list(map(lambda x: pd.read_csv(os.path.join(directory, x)), ['train.csv', 'test.csv']))
        train_x, train_y, test_x = load_fn(train_and_test_data)

        return train_x, train_y, test_x
    except Exception as e:
        print(e)
        return None, None, None
