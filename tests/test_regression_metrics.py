# -*- coding: utf-8 -*-
"""回归评估指标单元测试（学生C / 树模型体系）

运行方式（项目根目录）：
    pytest tests/test_regression_metrics.py -v
"""
import numpy as np
import pytest

from ml_engine.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


def test_perfect_prediction():
    """完美预测：MAE/MSE/RMSE 均为0，R²为1。"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.0)
    assert mean_squared_error(y_true, y_pred) == pytest.approx(0.0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(0.0)
    assert r2_score(y_true, y_pred) == pytest.approx(1.0)


def test_fixed_numeric_values():
    """固定数值用例，手工核对四个指标的计算结果。

    y_true = [3, 5, 7]，y_pred = [2, 5, 8]，误差 = [-1, 0, 1]：
      MAE  = (1+0+1)/3 = 2/3
      MSE  = (1+0+1)/3 = 2/3
      RMSE = sqrt(2/3)
      均值=5，SS_tot = 4+0+4 = 8，SS_res = 2
      R²   = 1 - 2/8 = 0.75
    """
    y_true = np.array([3.0, 5.0, 7.0])
    y_pred = np.array([2.0, 5.0, 8.0])

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    assert mean_squared_error(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
        np.sqrt(2.0 / 3.0))
    assert r2_score(y_true, y_pred) == pytest.approx(0.75)
    #RMSE 必须等于 MSE 的平方根
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
        np.sqrt(mean_squared_error(y_true, y_pred)))


def test_list_and_array_inputs_equal():
    """Python 列表与 numpy 数组两种输入的计算结果必须完全一致。"""
    y_true_list = [2.0, 4.0, 6.0, 8.0]
    y_pred_list = [1.5, 4.5, 5.5, 8.5]
    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)

    for fn in (mean_absolute_error, mean_squared_error,
               root_mean_squared_error, r2_score):
        assert fn(y_true_list, y_pred_list) == pytest.approx(
            fn(y_true_arr, y_pred_arr))


def test_constant_y_true_r2_no_error():
    """边界：y_true 全部相同导致 R² 分母为0时，不报错并返回0.0；
    其余三个误差指标仍正常计算。"""
    y_true = [5.0, 5.0, 5.0, 5.0]
    y_pred = [4.0, 5.0, 6.0, 5.0]

    assert r2_score(y_true, y_pred) == 0.0  # 不抛除零异常
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.5)
    assert mean_squared_error(y_true, y_pred) == pytest.approx(0.5)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
        np.sqrt(0.5))


def test_mean_baseline_prediction_r2_zero():
    """预测恒等于真实值均值时，R² 应为0（模型不比均值基线好）。"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())

    assert r2_score(y_true, y_pred) == pytest.approx(0.0)


def test_length_mismatch_raises():
    """真实值与预测值长度不一致应抛出 ValueError。"""
    with pytest.raises(ValueError):
        mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        r2_score(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
