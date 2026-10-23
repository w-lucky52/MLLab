import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred, y_score=None, class_labels=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    numeric_labels = np.unique(np.concatenate([y_true, y_pred]))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    cm = confusion_matrix(y_true, y_pred, labels=numeric_labels).tolist()

    if class_labels is not None and len(class_labels) == len(numeric_labels):
        labels_out = list(class_labels)
    else:
        labels_out = numeric_labels.tolist()

    # Per-class precision / recall / f1 / support, in the same order as
    # confusion_matrix rows/columns and class_labels.
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=numeric_labels, zero_division=0
    )
    per_class_metrics = []
    for idx in range(len(numeric_labels)):
        per_class_metrics.append(
            {
                "class": labels_out[idx],
                "precision": float(precisions[idx]),
                "recall": float(recalls[idx]),
                "f1": float(f1s[idx]),
                "support": int(supports[idx]),
            }
        )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "class_labels": labels_out,
        "per_class_metrics": per_class_metrics,
    }