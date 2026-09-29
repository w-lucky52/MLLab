# ml_engine/datasets/regression.py
from sklearn.datasets import load_diabetes

from .base import DatasetAdapter, DatasetData


class DiabetesAdapter(DatasetAdapter):
    name = "diabetes"
    description = "Diabetes: 442 samples, 10 features, 1 target"
    task_type = "regression"

    def load(self) -> DatasetData:
        bunch = load_diabetes()
        return DatasetData(
            X=bunch.data,
            y=bunch.target,
            task_type=self.task_type,
            feature_names=list(bunch.feature_names),
            target_names=["target"],
        )