# ml_engine/datasets/__init__.py
from .base import DatasetAdapter, DatasetData, split_dataset
from .classification import (
    BreastCancerAdapter, DigitsAdapter, IrisAdapter, WineAdapter,
)
from .regression import DiabetesAdapter

_DATASETS = {
    a.name: a
    for a in (
        IrisAdapter(), WineAdapter(), BreastCancerAdapter(),
        DigitsAdapter(), DiabetesAdapter(),
    )
}

def get_dataset(name: str) -> DatasetAdapter:
    if name not in _DATASETS:
        raise KeyError(name)
    return _DATASETS[name]

def list_datasets() -> list[DatasetAdapter]:
    return list(_DATASETS.values())