from unittest import TestCase
from titanic.data.preprocess import strategy_1, strategy_2
from titanic.data.load_dataset import load_titanic_csv_dataset
import pandas as pd
import os


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


class TestData(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_preprocess_strategy_1(self):
        dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
        train_and_test_data = list(map(lambda x: pd.read_csv(os.path.join(dataset_dir, x)), ['train.csv', 'test.csv']))
        train_x, train_y, test_x = strategy_1(train_and_test_data)

        print(train_x.info())
        print(train_x.head(2))
        print(train_x.columns.values)

    def test_load_titanic_csv_dataset(self):
        train_x, train_y, test_x = load_titanic_csv_dataset('/Users/hyeseong/datasets/private/kaggle/titanic', strategy='strategy_3')
        print(train_x.head(3))
        print(test_x.head(3))