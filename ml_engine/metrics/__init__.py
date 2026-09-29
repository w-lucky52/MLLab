# -*- coding: utf-8 -*-
"""模型评估指标模块（学生C / 树模型体系）。"""

from .regression import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

__all__ = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
]
