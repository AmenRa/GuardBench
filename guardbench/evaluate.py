import warnings

from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def false_positive_rate(fp: int, tn: int) -> float:
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def false_negative_rate(fn: int, tp: int) -> float:
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def evaluate(y_true: dict, y_pred_prob: dict, threshold: float = 0.5) -> dict:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    if not set(y_true) == set(y_pred_prob):
        raise ValueError("Different IDs!")

    # Ensure consistent sorting
    y_pred_prob = {x: y_pred_prob[x] for x in y_true}

    y_true = list(y_true.values())
    y_pred_prob = list(y_pred_prob.values())
    y_pred_binary = [x > threshold for x in y_pred_prob]

    p = sum(y_true)
    n = len(y_true) - sum(y_true)

    precision = precision_score(y_true, y_pred_binary, zero_division=0.0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0.0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0.0)

    if all(not y for y in y_true) and all(not y for y in y_pred_binary):
        auprc = 0.0
    else:
        auprc = average_precision_score(y_true, y_pred_prob)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred_binary, labels=[False, True]
    ).ravel()
    sensitivity = sensitivity_score(y_true, y_pred_binary)
    specificity = specificity_score(y_true, y_pred_binary)
    g_mean = geometric_mean_score(y_true, y_pred_binary)
    fpr = false_positive_rate(fp, tn)
    fnr = false_negative_rate(fn, tp)

    if all(y for y in y_true) and all(y for y in y_pred_binary):
        mcc = 0.0
    elif all(not y for y in y_true) and all(not y for y in y_pred_binary):
        mcc = 0.0
    else:
        mcc = matthews_corrcoef(y_true, y_pred_binary)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc) if n > 0 else 0.0,
        "auprc": float(auprc) if n > 0 else 0.0,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity) if n > 0 else 0.0,
        "g_mean": float(g_mean) if n > 0 else 0.0,
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "p": p,
        "n": n,
    }


def list_metrics():
    return [
        "precision",
        "recall",
        "f1",
        "mcc",
        "auprc",
        "sensitivity",
        "specificity",
        "g_mean",
        "fpr",
        "fnr",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
