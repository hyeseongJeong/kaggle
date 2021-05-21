from sklearn.tree import DecisionTreeClassifier
from titanic.data.load_dataset import load_titanic_csv_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report
from sklearn.tree import export_graphviz
import graphviz


def main():
    seed = 37
    dataset_dir = '/Users/hyeseong/datasets/private/kaggle/titanic'
    train_x, train_y, test_x = load_titanic_csv_dataset(dataset_dir, strategy='strategy_3')
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

    feature_names = train_x.columns.values

    model = DecisionTreeClassifier(max_depth=5,
                                   random_state=seed)
    model.fit(train_x, train_y)
    print(model.score(train_x, train_y))
    pred_y = model.predict(val_x)

    cm = confusion_matrix(y_true=val_y, y_pred=pred_y)
    ConfusionMatrixDisplay(cm, display_labels=('unsurvived', 'survived')).plot(values_format='.5g', cmap='Blues_r')

    print(classification_report(val_y, pred_y))
    #
    # export_graphviz(model,
    #                 feature_names=feature_names,
    #                 class_names=["Perish", "Survived"],
    #                 out_file="decision-tree.dot")
    #
    # with open("./decision-tree.dot") as f:
    #     dot_graph = f.read()
    #
    # graphviz.Source(dot_graph)


if __name__ == "__main__":
    main()