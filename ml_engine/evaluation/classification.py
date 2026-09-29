import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

def evaluate(y_true, y_pred, y_score=None, class_labels=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    numeric_labels = np.unique(np.concatenate([y_true, y_pred]))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=numeric_labels).tolist()

    if class_labels is not None and len(class_labels) == len(numeric_labels):
        labels_out = list(class_labels)
    else:
        labels_out = numeric_labels.tolist()

    return {"metrics": metrics, "confusion_matrix": cm, "class_labels": labels_out}