# -*- coding: utf-8 -*-
"""CartRegressionTree 单元测试（学生C / 树模型体系）

运行方式（项目根目录）：
    pytest tests/test_cart.py -v
"""
import numpy as np
import pytest

from ml_engine.cart import CartRegressionTree


def _make_regression_data(n_samples=50, n_features=3, seed=0):
    """构造多特征模拟回归数据：特征0强相关、特征1次相关、特征2基本无关。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n_samples, n_features))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.standard_normal(n_samples)
    return X, y


def test_predict_length_matches_samples():
    """拟合后预测输出长度应与输入样本数一致。"""
    X, y = _make_regression_data()
    model = CartRegressionTree(max_depth=3)
    model.fit(X, y)

    pred = model.predict(X)

    assert isinstance(pred, np.ndarray)
    assert pred.shape[0] == X.shape[0]


def test_feature_importances_shape_and_sum():
    """特征重要性长度等于特征维度，非负且总和近似为1。"""
    X, y = _make_regression_data()
    model = CartRegressionTree(max_depth=4)
    model.fit(X, y)

    importances = model.feature_importances_

    assert isinstance(importances, np.ndarray)
    assert importances.shape == (X.shape[1],)
    assert np.all(importances >= 0.0)
    assert pytest.approx(1.0, abs=1e-9) == float(importances.sum())


def test_single_sample_single_feature_runs():
    """极端场景：单一样本、单一特征也能正常拟合、预测并给出重要性。"""
    X = np.array([[5.0]])
    y = np.array([2.0])
    model = CartRegressionTree(max_depth=3)

    model.fit(X, y)  # 不应抛出任何异常
    pred = model.predict(X)

    assert pred.shape == (1,)
    assert model.feature_importances_.shape == (1,)
    # 无法分裂时不产生增益，重要性为0
    assert float(model.feature_importances_[0]) == 0.0


def test_constant_target_gives_zero_importances():
    """标签无波动时所有分裂增益为0，特征重要性应全部置0，预测恒为常数。"""
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.ones(10)
    model = CartRegressionTree(max_depth=3)

    model.fit(X, y)
    pred = model.predict(X)

    np.testing.assert_allclose(pred, np.ones(10))
    np.testing.assert_array_equal(model.feature_importances_, np.zeros(2))
