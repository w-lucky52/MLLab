# ml_engine/datasets/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetData:
    """统一的数据返回结构，D 的后端和 B/C 的模型都依赖此结构"""
    X: np.ndarray
    y: np.ndarray
    task_type: str  # "classification" / "regression"
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None


class DatasetAdapter(ABC):
    """所有数据集加载器的抽象基类"""
    name: str = ""
    description: str = ""
    task_type: str = ""  # "classification" / "regression"

    @abstractmethod
    def load(self) -> DatasetData:
        """返回 DatasetData 对象"""
        raise NotImplementedError


def split_dataset(X, y, test_size=0.3, random_state=42,
                  task_type="classification", standardize=True):
    if test_size not in (0.2, 0.3, 0.4):
        raise ValueError(f"test_size 只能是 0.2/0.3/0.4，收到: {test_size}")
    
    stratify = y if task_type == "classification" else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    return X_train, X_test, y_train, y_test