# ml_engine/evaluation/classification.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_classification(y_true, y_pred) -> dict:
    """
    分类任务评估指标，供 D 统一调度
    返回统一格式：
    {
        "metrics": {...},
        "confusion_matrix": [...],
        "class_labels": [...]
    }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 获取出现的所有类别标签
    labels = np.unique(np.concatenate([y_true, y_pred]))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "class_labels": labels.tolist(),
    }