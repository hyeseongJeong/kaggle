from titanic.data.load_dataset import load_titanic_csv_dataset
from titanic.dl.model.dense_net import dense_net

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from numpy.random import seed

import tensorflow as tf
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 200)


seed(0)
tf.random.set_seed(0)


def grid_search_cv(dataset):
    train_x, train_y = dataset
    n_features = train_x.shape[1]

    param_grid = {
        'activation': ["tanh", "relu"],
        'input_shape': [(n_features,),],
        'layer_dims': [(n_features,), (n_features,)*2, (n_features*2,), (n_features*2,) *2, (n_features*10,), (n_features*10,) *2],
        'batch_size': [16, 32,],
        'dropout_rate': [0.2, 0.4,],
    }

    model = KerasClassifier(dense_net, verbose=0)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_result = grid_search.fit(train_x, train_y, epochs=30)
    grid_results_df = pd.DataFrame(grid_search.cv_results_)

    return grid_result, grid_results_df


def main():
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    train_x, train_y, test_x = load_titanic_csv_dataset(dataset_dir, strategy='strategy_1')
    print(train_x.head(3))

    grid_result, grid_results_df = grid_search_cv((train_x, train_y))

    print(f"Best parameters: {grid_result.best_params_}")
    print(f"Best Crossvalidation score: {grid_result.best_score_:.3f}")

    print(grid_results_df[['params', 'mean_test_score']].sort_values(by='mean_test_score', ascending=False).head(10))
    grid_results_df.to_csv('/tmp/grid_results.csv', index=False)


if __name__ == "__main__":
    main()