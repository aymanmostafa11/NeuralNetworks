import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_dict=None, verbose=True):
    """
    Calculate Confusion matrix for binary labeled data

    :param y_true: Ground truth values
    :param y_pred: Predicted values
    :param label_dict: a dictionary containing 2 keys: "positive_label" and "negative_label", if not provided positive and negative labels are inferred from data
    :param verbose: print result after finishing
    :return: a dictionary containing tp, fp, tn, fn, precision and recall
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if label_dict:
        assert len(label_dict.keys()) == 2, "Current confusion matrix implementation supports binary labels only"
        assert "positive_label" in label_dict.keys() and "negative_label" in label_dict.keys(),\
            "label_dict must contain 'positive_label' and 'negative_label'"

        positive_label = label_dict['positive_label']
        negative_label = label_dict['negative_label']

    else:
        labels = np.unique(y_true)
        assert len(labels) == 2, "Current confusion matrix implementation supports binary labels only"
        positive_label = labels[0]
        negative_label = labels[1]

    tp = sum( (y_true == positive_label) & (y_true == y_pred) )
    tn = sum( (y_true == negative_label) & (y_true == y_pred) )
    fn = sum( (y_true == positive_label) & (y_true != y_pred) )
    fp = sum( (y_true == negative_label) & (y_true != y_pred) )

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = {'true_positive': tp,
              'true_negative': tn,
              'false_negative': fn,
              'false_positive': fp,
              'precision': precision,
              'recall': recall}

    if verbose:
        print(result)

    return result


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, verbose=True):
    acc = round( sum(y_true == y_pred) / len(y_true) * 100, 2)
    if verbose:
        print(acc)

    return acc
