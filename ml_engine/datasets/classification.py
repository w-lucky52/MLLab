# ml_engine/datasets/classification.py
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from .base import DatasetAdapter, DatasetData


class IrisAdapter(DatasetAdapter):
    name = "iris"
    description = "Iris: 150 samples, 4 features, 3 classes"
    task_type = "classification"

    def load(self) -> DatasetData:
        bunch = load_iris()
        return DatasetData(
            X=bunch.data,
            y=bunch.target,
            task_type=self.task_type,
            feature_names=list(bunch.feature_names),
            target_names=list(bunch.target_names),
        )


class WineAdapter(DatasetAdapter):
    name = "wine"
    description = "Wine: 178 samples, 13 features, 3 classes"
    task_type = "classification"

    def load(self) -> DatasetData:
        bunch = load_wine()
        return DatasetData(
            X=bunch.data,
            y=bunch.target,
            task_type=self.task_type,
            feature_names=list(bunch.feature_names),
            target_names=list(bunch.target_names),
        )


class BreastCancerAdapter(DatasetAdapter):
    name = "breast_cancer"
    description = "Breast Cancer: 569 samples, 30 features, 2 classes"
    task_type = "classification"

    def load(self) -> DatasetData:
        bunch = load_breast_cancer()
        return DatasetData(
            X=bunch.data,
            y=bunch.target,
            task_type=self.task_type,
            feature_names=list(bunch.feature_names),
            target_names=list(bunch.target_names),
        )


class DigitsAdapter(DatasetAdapter):
    name = "digits"
    description = "Digits: 1797 samples, 64 features, 10 classes"
    task_type = "classification"

    def load(self) -> DatasetData:
        bunch = load_digits()
        return DatasetData(
            X=bunch.data,
            y=bunch.target,
            task_type=self.task_type,
            feature_names=list(bunch.feature_names),
            target_names=[str(i) for i in range(10)],
        )