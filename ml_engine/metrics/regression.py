# -*- coding: utf-8 -*-
"""
回归评估指标集合（学生C / 树模型体系）

纯 numpy 实现 4 个标准回归指标，供算法自测与后端 D 同学统一封装调用：
    MAE  —— 平均绝对误差
    MSE  —— 均方误差
    RMSE —— 均方根误差
    R²   —— 决定系数
"""
import numpy as np


def _check_arrays(y_true, y_pred):
    """把输入统一转换为一维 float numpy 数组，并校验长度一致。

    支持 numpy 数组、Python list 等可被 np.asarray 转换的序列。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true 与 y_pred 长度不一致：{} != {}".format(
                y_true.shape[0], y_pred.shape[0]))
    if y_true.size == 0:
        raise ValueError("y_true 与 y_pred 不能为空")
    return y_true, y_pred


def mean_absolute_error(y_true, y_pred):
    """平均绝对误差 MAE（Mean Absolute Error）。

        MAE = mean(|y_true - y_pred|)

    含义：预测值与真实值偏差绝对值的平均，单位与标签相同，对异常值相对稳健；
    值越小拟合越好，完美预测时为 0。
    """
    y_true, y_pred = _check_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred):
    """均方误差 MSE（Mean Squared Error）。

        MSE = mean((y_true - y_pred)^2)

    含义：预测误差平方的平均，会放大大误差的惩罚；值越小拟合越好，
    完美预测时为 0。
    """
    y_true, y_pred = _check_arrays(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """均方根误差 RMSE（Root Mean Squared Error）。

        RMSE = sqrt(MSE)

    含义：MSE 的平方根，单位与标签相同，比 MSE 更直观；对大误差敏感，
    完美预测时为 0。
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred):
    """决定系数 R²（Coefficient of Determination）。

        R² = 1 - SS_res / SS_tot
        SS_res = sum((y_true - y_pred)^2)       # 残差平方和（模型误差）
        SS_tot = sum((y_true - mean(y_true))^2) # 总离差平方和（以均值为基线）

    含义：模型解释了多少比例的标签波动，取值通常在 (-inf, 1]：
    R²=1 为完美预测，R²=0 等价于直接用均值预测，负值说明比均值基线还差。
    边界：当 y_true 全部相同（SS_tot=0）时分母为 0，此时 R² 无定义，
    按约定返回 0.0，避免除零报错。
    """
    y_true, y_pred = _check_arrays(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        #真实标签无波动：R²无定义，按小组约定返回0.0
        return 0.0
    return float(1.0 - ss_res / ss_tot)
