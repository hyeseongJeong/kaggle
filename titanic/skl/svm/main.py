from sklearn.svm import SVC
from titanic.data.load_dataset import load_titanic_csv_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report

import joblib
import matplotlib.pyplot as plt


def main():
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    train_x, train_y, test_x = load_titanic_csv_dataset(dataset_dir, strategy='strategy_3')
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(train_x, train_y)
    print('train_acc: ', round(classifier.score(train_x, train_y) * 100, 2))
    pred_y = classifier.predict(val_x)

    cm = confusion_matrix(y_true=val_y, y_pred=pred_y)
    ConfusionMatrixDisplay(cm, display_labels=('unsurvived', 'survived')).plot(values_format='.5g', cmap='Blues_r')

    print(classification_report(val_y, pred_y))
    plt.show()

    joblib.dump(classifier, './classifier.pkl')


if __name__ == '__main__':
    main()