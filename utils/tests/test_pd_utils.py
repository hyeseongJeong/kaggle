from unittest import TestCase
from utils.pd_utils import *


class TestPdUtils(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_profile_report(self):
        titanic_train_csv_file_path = '/Users/hyeseong/datasets/private/kaggle/titanic/train.csv'
        data = pd.read_csv(titanic_train_csv_file_path)
        profile_report(data, title='train_csv_profile_report')