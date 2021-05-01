from sklearn.ensemble import RandomForestClassifier
from titanic.data.load_dataset import load_titanic_csv_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report

import joblib
import matplotlib.pyplot as plt


def main():
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    train_x, train_y, test_x = load_titanic_csv_dataset(dataset_dir, strategy='strategy_2')
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=2)
    model.fit(train_x, train_y)

    # 기존에 저장한 모델을 불러와서 사용할 수도 있음
    # model = joblib.load('model.pkl')

    pred_y = model.predict(val_x)

    cm = confusion_matrix(y_true=val_y, y_pred=pred_y)
    ConfusionMatrixDisplay(cm, display_labels=('unsurvived', 'survived')).plot(values_format='.5g', cmap='Blues_r')

    print(classification_report(val_y, pred_y))
    plt.show()

    joblib.dump(model, './model.pkl')


if __name__ == "__main__":
    main()
