import numpy as np
from beautifultable import BeautifulTable
import sklearn.metrics


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_names_dict = None, label_values_dict=None, verbose=True):
    """
    Calculate Confusion matrix for binary labeled data

    :param y_true: Ground truth values
    :param y_pred: Predicted values
    :param label_names_dict: a dictionary containing a key for "pos" and "neg" for classes names (for printing table)
    :param label_values_dict: a dictionary containing 2 keys: "pos" and "neg", if not provided
    positive and negative labels are inferred from data
    :param verbose: print result after finishing
    :return: a dictionary containing tp, fp, tn, fn, precision and recall

    :Example: confusion_matrix(y, predicted, {"positive_label": "Survived", "negative_label": "Didn't Survive"},
                                {"positive_label": 1, "negative_label": 1}

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if label_values_dict:
        assert len(label_values_dict.keys()) == 2, "Current confusion matrix implementation supports binary labels only"
        assert "pos" in label_values_dict.keys() and "neg" in label_values_dict.keys(),\
            "label_dict must contain 'pos' and 'neg'"

        positive_label = label_values_dict['pos']
        negative_label = label_values_dict['neg']

    else:
        labels = np.unique(y_true)
        assert len(labels) == 2, "Current confusion matrix implementation supports binary labels only"
        positive_label = labels[0]
        negative_label = labels[1]

    positive_label_name = ""
    negative_label_name = ""
    if label_names_dict:
        positive_label_name = label_names_dict["pos"]
        negative_label_name = label_names_dict["neg"]

    tp = np.sum( (y_true == positive_label) & (y_true == y_pred) )
    tn = np.sum( (y_true == negative_label) & (y_true == y_pred) )
    fn = np.sum( (y_true == positive_label) & (y_true != y_pred) )
    fp = np.sum( (y_true == negative_label) & (y_true != y_pred) )

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = {'true_positive': tp,
              'true_negative': tn,
              'false_negative': fn,
              'false_positive': fp,
              'precision': precision,
              'recall': recall}

    if verbose:
        print("\n\n##########################")
        print("#### Confusion Matrix ####")
        print("##########################")
        table = BeautifulTable()
        table.columns.header = ["", f'Actual Positive (P)\n"{positive_label_name}"',
                                f'Actual Negative (N)\n"{negative_label_name}"']
        table.rows.append([f'Predicted Positive (P)\n"{positive_label_name}"', f"{tp} (TP)", f"{fp} (FP)"])
        table.rows.append([f'Predicted Negative (N)\n "{negative_label_name}"', f"{fn} (FN)", f"{tn} (TN)"])
        print(table)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    return result


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, verbose=False):
    acc = np.round( np.sum(y_true == y_pred) / len(y_true) * 100, 2)
    if verbose:
        print(acc)

    return acc

def confusion_matrix_for_multiclass(y_true, y_pred):
    conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
    print("\n\n##########################")
    print("#### Confusion Matrix ####")
    print("##########################")
    table = BeautifulTable()
    for row in range(0,conf_mat.shape[0]):
        table.append_row(conf_mat[row])
    print(table)
    
#
# a = np.array([1,0,0,1])
# b = np.array([1,0,1,0])
#
# confusion_matrix(b, a, {"pos": "pos", "neg": "neg"}, {"pos":1, "neg": 0})