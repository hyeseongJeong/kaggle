import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


def strategy_1(train_and_test_dataset: [pd.DataFrame, pd.DataFrame]):
    scaler = None

    for i, _d in enumerate(train_and_test_dataset):
        # STEP 1. CREATE NEW FEATURES
        _d['Title'] = _d['Name'].str.extract('([A-Za-z]+)\.', expand=False)

        for a, b in [[['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'],
                     ['Mlle', 'Miss'],
                     ['Ms', 'Miss'],
                     ['Mme', 'Mrs']]:
            _d['Title'] = _d['Title'].replace(a, b)

        _d['FamilySize'] = _d['Parch'] + _d['SibSp']

        # STEP 2. REPLACE NAN VALUES
        mask = _d.Age.isnull()
        title_mean_age = _d[['Title', 'Age']].groupby(['Title'], as_index=True).mean()
        title_mean_age_dict = title_mean_age.to_dict()['Age']
        _d.loc[mask, 'Age'] = _d.loc[mask, 'Title'].map(title_mean_age_dict)

        _d.Embarked.fillna(_d.Embarked.mode().iloc[0], inplace=True)

        # STEP 3. CONVERT CATEGORICAL FEATURES
        _d = pd.get_dummies(_d, columns=['Title', 'Embarked', 'Pclass'], drop_first=False)  # add "drop_first=True" option if you want to avoid multicollinearity.

        # STEP 4. FEATURE SCALING
        features = ['Age', 'FamilySize']
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(_d[features])
        _d[features] = scaler.transform(_d[features])

        # STEP 5. ORDINAL ENCODING
        sex_keymap = {'female': 1, 'male': 0}
        _d['Sex'] = _d['Sex'].map(sex_keymap).astype(int)

        # STEP 6. DROP features
        drop_features = ['PassengerId', 'Cabin', 'Name', 'Ticket', 'Parch', 'SibSp']
        _d.drop(drop_features, inplace=True, axis='columns')

        train_and_test_dataset[i] = _d

    train_xy, test_x = train_and_test_dataset
    train_x, train_y = train_xy.drop(['Survived'], axis=1), train_xy['Survived']

    return train_x, train_y, test_x


def strategy_2(train_and_test_dataset: [pd.DataFrame, pd.DataFrame]):
    train_df, test_df = train_and_test_dataset
    # Convert categorical data to dummy encording
    train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # Create a new feature (FamilySize)
    train_df['FamilySize'] = train_df['Parch'] + train_df['SibSp']
    test_df['FamilySize'] = test_df['Parch'] + test_df['SibSp']

    # Create a new feature (Title)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                     'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # Replace missing data
    # Age (train_df, test_df)
    df_concat = pd.concat([train_df, test_df])
    title_age_mean = df_concat[['Title', 'Age']].groupby(['Title'], as_index=True).mean()
    mapping_dict = title_age_mean.to_dict()['Age']
    mask = train_df['Age'].isna()
    train_df.loc[mask, 'Age'] = train_df.loc[mask, 'Title'].map(mapping_dict)
    mask = test_df['Age'].isna()
    test_df.loc[mask, 'Age'] = test_df.loc[mask, 'Title'].map(mapping_dict)
    # Embarked
    train_df['Embarked'].fillna(train_df['Embarked'].mode().iloc[0], inplace=True)

    # Converting categorical features
    # add "drop_first=True" option to avoid multicollinearity.
    train_df = pd.get_dummies(train_df, columns=['Title', 'Embarked'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Title', 'Embarked'], drop_first=True)

    # Feature scaling
    #features = ['Age', 'SibSp', 'Parch']
    features = ['Age', 'FamilySize']
    scaler = MinMaxScaler()
    ##scaler = StandardScaler()
    scaler.fit(train_df[features])
    train_df[features] = scaler.transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    # Drop features
    drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Fare', 'SibSp', 'Parch']
    X_train = train_df.drop(drop_features + ['Survived'], axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop(drop_features, axis=1)

    return X_train, y_train, X_test


def strategy_3(train_and_test_dataset: [pd.DataFrame, pd.DataFrame]):
    scaler = None

    for i, _d in enumerate(train_and_test_dataset):
        # STEP 1. CREATE NEW FEATURES
        _d['Title'] = _d['Name'].str.extract('([A-Za-z]+)\.', expand=False)

        for a, b in [[['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'],
                     ['Mlle', 'Miss'],
                     ['Ms', 'Miss'],
                     ['Mme', 'Mrs']]:
            _d['Title'] = _d['Title'].replace(a, b)

        _d['FamilySize'] = _d['Parch'] + _d['SibSp']

        # STEP 2. REPLACE NAN VALUES
        mask = _d.Age.isnull()
        title_mean_age = _d[['Title', 'Age']].groupby(['Title'], as_index=True).mean()
        title_mean_age_dict = title_mean_age.to_dict()['Age']
        _d.loc[mask, 'Age'] = _d.loc[mask, 'Title'].map(title_mean_age_dict)

        _d.Embarked.fillna(_d.Embarked.mode().iloc[0], inplace=True)

        _d = _d.fillna(_d['Fare'].mean())

        # Step 3. Convert Continuous Features to Categorical
        # # 방법이 여러가지라 섞어서 써봄.
        _d['Fare_bin'] = 0
        _d.loc[_d.Fare <= 7.91, 'Fare_bin'] = 0
        _d.loc[(_d.Fare > 7.91) & (_d.Fare <= 14.454), 'Fare_bin'] = 1
        _d.loc[(_d.Fare > 14.454) & (_d.Fare <= 31.0), 'Fare_bin'] = 2
        _d.loc[(_d.Fare > 31.0) & (_d.Fare <= 512.329), 'Fare_bin'] = 3

        _d['Age_bin'] = pd.cut(_d['Age'],
                               bins=[0, 12, 20, 40, 120],
                               labels=['Children', 'Teenage', 'Adult', 'Elder'])

        _d['FamilySize_bin'] = 0
        _d.loc[(_d.FamilySize >= 1) & (_d.FamilySize <= 3), 'FamilySize_bin'] = 1
        _d.loc[_d.FamilySize > 3, 'FamilySize_bin'] = 2

        # STEP 4. CONVERT CATEGORICAL FEATURES
        for _f in ['Title', 'Embarked', 'Pclass', 'Sex']:
            le = LabelEncoder()
            _d[_f] = le.fit_transform(_d[_f])
        _d = pd.get_dummies(_d, columns=['Sex', 'Title', 'Embarked', 'Pclass', 'Fare_bin', 'FamilySize_bin', 'Age_bin'], drop_first=False)  # add "drop_first=True" option if you want to avoid multicollinearity.

        # STEP 5. DROP features
        drop_features = ['PassengerId', 'Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Fare', 'FamilySize', 'Age']
        _d.drop(drop_features, inplace=True, axis='columns')

        train_and_test_dataset[i] = _d

    train_xy, test_x = train_and_test_dataset
    train_x, train_y = train_xy.drop(['Survived'], axis=1), train_xy['Survived']

    return train_x, train_y, test_x


if __name__ == "__main__":
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    from utils.visualization_utils import show_facet_grid, show_bar_plot, show_facet_grid2
    from utils.pd_utils import get_x_rate_by_group

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    train_and_test_data = list(map(lambda x: pd.read_csv(os.path.join('/Users/hyeseong/datasets/private/kaggle/titanic', x)), ['train.csv', 'test.csv']))
    train_x, train_y, test_x = strategy_3(train_and_test_data)
    print(train_x.head())
