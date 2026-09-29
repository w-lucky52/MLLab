# ml_engine/models/knn.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class KNNModel(BaseEstimator, ClassifierMixin):
    """KNN 分类器，包装 sklearn，统一接口 fit/predict"""
    # 参数元信息，供 D 的注册器读取，前端动态表单会用到
        parameters_metadata = {
        "n_neighbors": {
            "label": "邻居数", 
            "type": "int", 
            "min": 1, 
            "max": 20, 
            "step": 1, 
            "default": 5, 
            "description": "选择最近邻居的数量"
        },
        "weights": {
            "label": "权重方式", 
            "type": "select", 
            "options": ["uniform", "distance"], 
            "default": "uniform", 
            "description": "预测时邻居的权重方式"
        },
        "p": {
            "label": "距离度量", 
            "type": "int", 
            "min": 1, 
            "max": 2, 
            "step": 1, 
            "default": 2, 
            "description": "1为曼哈顿距离，2为欧氏距离"
        },
    }

    def __init__(self, n_neighbors=5, weights="uniform", p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self._model = None
        self.classes_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        self._model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, weights=self.weights, p=self.p
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._model.predict(X)