from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report
from titanic.data.load_dataset import load_titanic_csv_dataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tensorflow as tf


def main():
    # MODEL BUILD
    model = tf.keras.models.load_model('./dl/ckpt/')
    model.summary()

    # DATASET
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    X_train, y_train, X_test = load_titanic_csv_dataset(dataset_dir, strategy='strategy_1')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    y_prob = model.predict(x=X_val)
    y_pred = (y_prob > 0.5) * 1

    cm = confusion_matrix(y_true=y_val, y_pred=y_pred)
    ConfusionMatrixDisplay(cm, display_labels=('unsurvived', 'survived')).plot(values_format='.5g', cmap='Blues_r')

    print(classification_report(y_val, y_pred))
    plt.show()


if __name__ == "__main__":
    main()