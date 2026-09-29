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


def split_dataset(
    data: DatasetData,
    test_size: float = 0.3,
    standardize: bool = True,
    random_state: int = 42,
):
    """
    防数据泄漏切分：
    1. 先切分 train / test
    2. 只在训练集上 fit Scaler
    3. 只标准化 X，不动 y
    4. 分类任务使用 stratify=y
    """
    if test_size not in (0.2, 0.3, 0.4):
        raise ValueError(f"test_size 只能是 0.2/0.3/0.4，收到: {test_size}")

    stratify = data.y if data.task_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)   # 只在训练集 fit
        X_test = scaler.transform(X_test)          # 用训练集的参数变换测试集

    return X_train, X_test, y_train, y_test, scaler