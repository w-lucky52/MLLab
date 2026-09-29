# -*- coding: utf-8 -*-
"""RandomForestRegressor 单元测试（学生C / 树模型体系，P1）

运行方式（项目根目录）：
    pytest tests/test_random_forest.py -v
"""
import numpy as np
import pytest

from ml_engine.cart import CartRegressionTree
from ml_engine.random_forest import RandomForestRegressor


def _make_regression_data(n_samples=60, n_features=4, seed=0):
    """构造多特征模拟回归数据：特征0强相关、特征1次相关、其余特征无关。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n_samples, n_features))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.standard_normal(n_samples)
    return X, y


def test_predict_length_matches_samples():
    """拟合后预测输出长度应与输入样本数一致，集成输出为numpy数组。"""
    X, y = _make_regression_data()
    model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=0)
    model.fit(X, y)

    pred = model.predict(X)

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (X.shape[0],)


def test_feature_importances_shape_and_sum():
    """特征重要性长度等于特征维度，非负且总和近似为1。"""
    X, y = _make_regression_data()
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=1)
    model.fit(X, y)

    importances = model.feature_importances_

    assert isinstance(importances, np.ndarray)
    assert importances.shape == (X.shape[1],)
    assert np.all(importances >= 0.0)
    assert pytest.approx(1.0, abs=1e-9) == float(importances.sum())


def test_n_estimators_one_equals_single_cart():
    """n_estimators=1、max_features=None 时，森林预测应与“同一bootstrap样本上
    训练的单棵 CartRegressionTree”完全一致，验证平均集成逻辑与基学习器复用。"""
    X, y = _make_regression_data()
    rf = RandomForestRegressor(
        n_estimators=1, max_depth=4, max_features=None, random_state=2)
    rf.fit(X, y)

    #基学习器必须就是 CartRegressionTree（100%复用，不另起炉灶）
    assert len(rf.estimators_) == 1
    assert isinstance(rf.estimators_[0], CartRegressionTree)

    #用森林记录的同一组bootstrap下标训练一棵原生CART，结果应完全相同
    boot_idx = rf.estimators_samples_[0]
    single = CartRegressionTree(max_depth=4)
    single.fit(X[boot_idx], y[boot_idx])

    #单棵树时，森林的平均集成预测就是这棵树自己的预测
    np.testing.assert_allclose(rf.predict(X), single.predict(X))


def test_single_feature_few_samples_runs():
    """极端场景：单特征、少量样本可正常拟合、预测并给出重要性，不抛异常。"""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0, 2.1, 2.9, 4.2, 5.0, 6.3])
    model = RandomForestRegressor(
        n_estimators=5, max_depth=2, max_features="sqrt", random_state=3)

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (6,)
    assert model.feature_importances_.shape == (1,)
    #该数据每棵树都能正常分裂，重要性应为1
    assert pytest.approx(1.0, abs=1e-9) == float(
        model.feature_importances_[0])


def test_random_state_reproducible():
    """相同 random_state 两次训练，bootstrap/特征采样与预测应完全一致。"""
    X, y = _make_regression_data()

    m1 = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=42)
    m2 = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=42)
    m1.fit(X, y)
    m2.fit(X, y)

    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
    np.testing.assert_array_equal(
        m1.feature_importances_, m2.feature_importances_)


@pytest.mark.parametrize("max_features", ["sqrt", "log2", None, 2, 0.5])
def test_max_features_variants_run(max_features):
    """max_features 的五种取值形式都应被正确解析并正常训练预测。"""
    X, y = _make_regression_data(n_features=4)
    model = RandomForestRegressor(
        n_estimators=4, max_depth=3, max_features=max_features, random_state=0)

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (X.shape[0],)
    assert pytest.approx(1.0, abs=1e-9) == float(
        model.feature_importances_.sum())


def test_invalid_max_features_raises():
    """非法 max_features 应抛出 ValueError。"""
    X, y = _make_regression_data(n_features=4)
    model = RandomForestRegressor(max_features="bad_option")
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_predict_before_fit_raises():
    """未训练直接预测应抛出异常，避免静默返回错误结果。"""
    X, _ = _make_regression_data()
    with pytest.raises(ValueError):
        RandomForestRegressor(n_estimators=3).predict(X)
