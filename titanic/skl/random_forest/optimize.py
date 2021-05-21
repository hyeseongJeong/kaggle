from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from titanic.data.load_dataset import load_titanic_csv_dataset
from sklearn.model_selection import train_test_split


def main():
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    train_x, train_y, test_x = load_titanic_csv_dataset(dataset_dir, strategy='strategy_3')
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

    # Random Forest Classifier Parameters tunning
    model = RandomForestClassifier()
    n_estim = range(100, 1000, 100)
    max_depth = range(1, 10, 1)
    random_state = (2, )

    ## Search grid for optimal parameters
    param_grid = {"n_estimators": n_estim,
                  "max_depth": max_depth,
                  "random_state": random_state}

    gscv = GridSearchCV(model,
                        param_grid=param_grid,
                        cv=5,
                        scoring="accuracy",
                        n_jobs=4,
                        verbose=1)

    gscv.fit(train_x, train_y)

    # Best score
    print(gscv.best_score_)

    #best estimator
    print(gscv.best_estimator_)


if __name__ == "__main__":
    main()