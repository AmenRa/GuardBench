import warnings

from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def false_positive_rate(fp: int, tn: int) -> float:
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def false_negative_rate(fn: int, tp: int) -> float:
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def evaluate(y_true: dict, y_pred_prob: dict) -> dict:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    if not set(y_true) == set(y_pred_prob):
        raise ValueError("Different IDs!")

    # Ensure consistent sorting
    y_pred_prob = {x: y_pred_prob[x] for x in y_true}

    y_true = list(y_true.values())
    y_pred_prob = list(y_pred_prob.values())
    y_pred_binary = [x > 0.5 for x in y_pred_prob]

    precision = precision_score(y_true, y_pred_binary, zero_division=0.0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0.0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0.0)
    auprc = average_precision_score(y_true, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred_binary, labels=[False, True]
    ).ravel()
    sensitivity = sensitivity_score(y_true, y_pred_binary)
    specificity = specificity_score(y_true, y_pred_binary)
    g_mean = geometric_mean_score(y_true, y_pred_binary)
    fpr = false_positive_rate(fp, tn)
    fnr = false_negative_rate(fn, tp)

    return {
        "precision": precision if tn > 0 else 0.0,
        "recall": recall,
        "f1": f1,
        "auprc": auprc if tn > 0 else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "g_mean": g_mean,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
