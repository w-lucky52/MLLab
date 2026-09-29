# ml_engine/models/naive_bayes.py
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianNBModel(BaseEstimator, ClassifierMixin):
    """高斯朴素贝叶斯分类器，统一接口 fit/predict"""
    parameters_metadata = {
        "var_smoothing": {
            "label": "方差平滑", 
            "type": "float", 
            "min": 1e-12, 
            "max": 1e-1, 
            "step": 1e-10, 
            "default": 1e-9, 
            "description": "防止方差为零的平滑参数"
        },
    }

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self._model = None
        self.classes_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        self._model = GaussianNB(var_smoothing=self.var_smoothing)
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._model.predict(X)