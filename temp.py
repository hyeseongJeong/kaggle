import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


if __name__ == "__main__":
    submission_csv_path_1 = '/Users/hyeseong/projects/kaggle/titanic/dl/submission.csv'
    submission_csv_path_2 = '/Users/hyeseong/projects/kaggle/titanic/skl/random_forest/submission.csv'

    dl_result = pd.read_csv(submission_csv_path_1)
    rf_result = pd.read_csv(submission_csv_path_2)

    result_df = submission = pd.DataFrame({
        'PassengerId': dl_result["PassengerId"],
        'Survived_dl': dl_result["Survived"],
        'Survived_rf': rf_result["Survived"]
    })
    # print(result_df.info())
    print(result_df.loc[result_df['Survived_dl'] != result_df['Survived_rf']].count())