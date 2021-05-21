from sklearn.ensemble import RandomForestClassifier
from titanic.data.load_dataset import load_titanic_csv_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report

import joblib
import matplotlib.pyplot as plt


def main():
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    train_x, train_y, test_x = load_titanic_csv_dataset(dataset_dir, strategy='strategy_3')
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=2)
    # model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                                max_depth=None, max_features='auto', max_leaf_nodes=None,
    #                                min_impurity_decrease=0.0, min_impurity_split=None,
    #                                min_samples_leaf=1, min_samples_split=2,
    #                                min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
    #                                oob_score=False, random_state=None, verbose=0,
    #                                warm_start=False)
    model.fit(train_x, train_y)
    print('train score: ', model.score(train_x, train_y))

    # 기존에 저장한 모델을 불러와서 사용할 수도 있음
    # model = joblib.load('model.pkl')

    pred_y = model.predict(val_x)

    cm = confusion_matrix(y_true=val_y, y_pred=pred_y)
    ConfusionMatrixDisplay(cm, display_labels=('unsurvived', 'survived')).plot(values_format='.5g', cmap='Blues_r')

    print(classification_report(val_y, pred_y))
    # plt.show()

    joblib.dump(model, './model.pkl')


if __name__ == "__main__":
    main()
