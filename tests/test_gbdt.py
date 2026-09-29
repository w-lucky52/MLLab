# -*- coding: utf-8 -*-
"""GradientBoostingRegressor 单元测试（学生C / 树模型体系，P2）

运行方式（项目根目录）：
    pytest tests/test_gbdt.py -v
"""
import numpy as np
import pytest

from ml_engine.cart import CartRegressionTree
from ml_engine.gbdt import GradientBoostingRegressor


def _make_regression_data(n_samples=60, n_features=4, seed=0):
    """构造多特征模拟回归数据：特征0强相关、特征1次相关、其余特征无关。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n_samples, n_features))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.standard_normal(n_samples)
    return X, y


def test_predict_length_matches_samples():
    """拟合后预测输出长度应与输入样本数一致，集成输出为numpy数组。"""
    X, y = _make_regression_data()
    model = GradientBoostingRegressor(
        n_estimators=20, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    pred = model.predict(X)

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (X.shape[0],)


def test_feature_importances_shape_and_sum():
    """特征重要性长度等于特征维度，非负且总和近似为1。"""
    X, y = _make_regression_data()
    model = GradientBoostingRegressor(n_estimators=20, max_depth=3)
    model.fit(X, y)

    importances = model.feature_importances_

    assert isinstance(importances, np.ndarray)
    assert importances.shape == (X.shape[1],)
    assert np.all(importances >= 0.0)
    assert pytest.approx(1.0, abs=1e-9) == float(importances.sum())


def test_one_estimator_lr_one_equals_residual_tree():
    """n_estimators=1、learning_rate=1.0 时：
    预测 = 标签均值 + 一棵直接拟合残差(y-均值)的CART预测。
    利用“均值 + 叶内残差均值 = 叶内y均值”，结果应与直接用同深度CART拟合y
    完全一致，验证残差拟合与累加预测逻辑。"""
    X, y = _make_regression_data()
    gbr = GradientBoostingRegressor(
        n_estimators=1, learning_rate=1.0, max_depth=3)
    gbr.fit(X, y)

    #初始基准必须是标签均值
    assert gbr.init_value_ == pytest.approx(float(np.mean(y)))

    #直接用同深度原生CART拟合y，预测应逐元素相等
    direct_tree = CartRegressionTree(max_depth=3)
    direct_tree.fit(X, y)

    np.testing.assert_allclose(gbr.predict(X), direct_tree.predict(X))
    #基学习器数量为1且就是 CartRegressionTree（100%复用）
    assert len(gbr.estimators_) == 1
    assert isinstance(gbr.estimators_[0], CartRegressionTree)


def test_single_feature_few_samples_runs():
    """极端场景：单特征、少量样本可正常拟合、预测并给出重要性，不抛异常。"""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0, 2.1, 2.9, 4.2, 5.0, 6.3])
    model = GradientBoostingRegressor(
        n_estimators=5, learning_rate=0.5, max_depth=2)

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (6,)
    assert model.feature_importances_.shape == (1,)
    assert pytest.approx(1.0, abs=1e-9) == float(
        model.feature_importances_[0])


def test_training_loss_monotonically_decreases():
    """逐轮沿残差方向更新，训练MSE曲线应单调不增（允许浮点误差）。"""
    X, y = _make_regression_data()
    model = GradientBoostingRegressor(
        n_estimators=30, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    scores = model.train_score_

    assert scores.shape == (30,)
    assert np.all(np.diff(scores) <= 1e-12)
    #末轮MSE应显著低于初始常数预测的MSE
    assert scores[-1] < np.mean((y - np.mean(y)) ** 2)


def test_constant_target_zero_importances():
    """标签为常数时残差恒为0，所有树都不分裂：初始值=该常数、
    预测恒等于常数、特征重要性全为0。"""
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.full(10, 3.5)
    model = GradientBoostingRegressor(n_estimators=10, max_depth=3)

    model.fit(X, y)

    np.testing.assert_allclose(model.predict(X), np.full(10, 3.5))
    np.testing.assert_array_equal(model.feature_importances_, np.zeros(2))
    assert model.init_value_ == pytest.approx(3.5)


def test_predict_before_fit_raises():
    """未训练直接预测应抛出异常。"""
    X, _ = _make_regression_data()
    with pytest.raises(ValueError):
        GradientBoostingRegressor(n_estimators=3).predict(X)


@pytest.mark.parametrize("n_estimators,learning_rate",
                         [(0, 0.1), (5, 0.0), (5, -0.2)])
def test_invalid_params_raise(n_estimators, learning_rate):
    """非法超参应抛出 ValueError。"""
    X, y = _make_regression_data()
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate)
    with pytest.raises(ValueError):
        model.fit(X, y)
