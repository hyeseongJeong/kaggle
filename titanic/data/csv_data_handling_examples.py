"""
https://www.kaggle.com/kodamap/titanic-with-tensorflow-keras

https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""

# data analysis and wrangling

import os
import pandas as pd
import numpy as np
import random as rnd
from pprint import pprint

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)


def show_facet_grid(data: pd.DataFrame, col: str, sub_col: str, hue: str, aplha: float=.5, bins: int=20):
    g = sns.FacetGrid(data, col=col, hue=hue)
    g.map(sns.histplot, sub_col, alpha=aplha, bins=bins)
    g.add_legend()
    plt.show()


def get_x_rate_by_group(data:pd.DataFrame, x: str, group: str):

    return data[[group, x]].groupby([group], as_index=False).mean().sort_values(by=x, ascending=False)


def create_new_feature(data: pd.DataFrame, feature_name: str, condition: str):
    data[feature_name] = data.Name.str.extract(condition, expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')


def main():
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    combine = [train_df, test_df]

    # 데이터셋 살펴보기

    # print(train_df.info())
    # print(train_df.columns.values)
    # print(train_df.describe(include='all'))

    # Nan 데이터 확인하기

    # print(train_df.Age.isnull().value_counts())

    # 데이터 특성 분석

    # show_facet_grid(train_df, 'Sex', 'Age', 'Survived')
    # print(get_x_rate_by_group(train_df, "Survived", "Embarked"))
    # show_facet_grid(train_df, 'Embarked', 'Age', 'Survived')
    # print(get_x_rate_by_group(train_df, "Survived", "Pclass"))
    # show_facet_grid(train_df, 'Sex', 'Fare', 'Survived')
    # print(train_df.Cabin)

    # train_df.loc[train_df.Sex=='male', ['Age', 'Survived']].groupby('Age').mean().sort_index().plot(kind='bar')
    # train_df.loc[train_df.Sex=='female', ['Age', 'Survived']].groupby('Age').mean().sort_index().plot(kind='bar')
    # show_facet_grid(train_df, 'Age', 'Sex', 'Survived')
    #
    # plt.show()

    # new feature 생성

    for _d in train_df, test_df:
        create_new_feature(_d, 'Title', '([A-Za-z]+)\.')
        _d['FamilySize'] = _d['Parch'] + _d['SibSp']

    # print(pd.crosstab(train_df.Title, train_df.Sex))
    # print(get_x_rate_by_group(train_df, 'Survived', 'Title'))

    # show_facet_grid(train_df, 'Title', 'Age', 'Sex')
    # show_facet_grid(train_df, 'Title', 'Age', 'Survived')

    # print(get_x_rate_by_group(train_df, 'Survived', 'FamilySize'))

    # Replacing missing value

    mask = train_df.Age.isnull()
    # print(train_df[['Title', 'Age']].groupby(['Title'], as_index=True).describe())
    title_mean_age = train_df[['Title', 'Age']].groupby(['Title'], as_index=True).mean()
    title_mean_age_dict = title_mean_age.to_dict()['Age']
    # print(title_mean_age_dict)
    train_df.loc[mask, 'Age'] = train_df.loc[mask, 'Title'].map(title_mean_age_dict)
    # print(train_df[['PassengerId', 'Age', 'Title']].loc[mask])

    # print(train_df.Embarked.value_counts())
    # print(train_df.Embarked.mode().iloc[0])
    mask = train_df.Embarked.isnull()
    train_df.Embarked.fillna(train_df.Embarked.mode().iloc[0], inplace=True)
    # print(train_df.Embarked.loc[mask])
    # print(train_df.isnull().sum())

    # Converting categorical features
    # print(train_df.columns.values)
    # print(train_df.Title.unique())
    # print(train_df.Embarked.unique())
    # print(train_df.Sex.unique())
    # print(train_df.info())
    train_df = pd.get_dummies(train_df, columns=['Title', 'Embarked'])
    test_df = pd.get_dummies(test_df, columns=['Title', 'Embarked'])

    # Ordinal encoding (Sex)
    Ordinal_dict_of_sex = {'female': 1, 'male': 0}
    for _d in [train_df, test_df]:
        _d['Sex'] = _d['Sex'].map(Ordinal_dict_of_sex).astype(int)

    # feature_scaling
    train_df_non_scaled = train_df.copy()
    test_df_non_scaled = test_df.copy()
    features = ['Age', 'Fare']
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])
    train_df[features] = scaler.transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    # check scaled
    # fig, axes = plt.subplots(1,2, figsize=(12,4))
    # axes[0].set_title('Original')
    # axes[0].scatter(data=train_df_non_scaled, x="Age", y="Sex", marker="o", label="train_df_non_scaled")
    # axes[0].scatter(data=test_df_non_scaled, x="Age", y="Sex", marker="^", label="test_df_non_scaled")
    # axes[0].legend(loc='upper right')
    # axes[1].set_title('Scaled')
    # axes[1].scatter(data=train_df, x="Age", y="Sex", marker="o", label="train_df_scaled")
    # axes[1].scatter(data=test_df, x="Age", y="Sex", marker="^", label="test_df_scaled")
    # axes[1].legend(loc='upper right')
    # plt.show()

    # Remove features
    drop_features = ['PassengerId', 'Cabin', 'Name', 'Ticket', 'Parch', 'SibSp']
    for _d in [train_df, test_df]:
        _d.drop(drop_features, inplace=True, axis='columns')

    train_X = train_df.drop(['Survived'], axis=1)
    train_Y = train_df[['Survived']]
    test_X = test_df

    print(train_X.info(), '\n')
    print(train_Y.info(), '\n')
    print(test_X.info(), '\n')


if __name__ == "__main__":
    main()

