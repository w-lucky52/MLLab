"""Model registry and ModelAdapter implementations.

This module is the integration glue between the backend (which calls
:func:`get_model` / :func:`list_models`) and the concrete model classes
provided by the algorithm teammates.

Registration policy (agreed with B/C)
-------------------------------------
* Only models that are **implemented and merged to develop** are registered
  here. There is intentionally **no** ``except ImportError: pass``: a wrong
  class name, a broken dependency or a circular import must fail loudly
  instead of silently disappearing from the catalog.
* When a teammate's model branch is merged, add an explicit import and a
  registry entry below.

Parameter metadata
------------------
Each entry provides ``parameters`` (frontend form metadata). The adapter
derives ``default_parameters`` automatically from each parameter's
``default`` value, so the two never drift apart.

``type`` is one of ``int`` / ``float`` / ``bool`` / ``select``. Numeric
parameters may carry ``min`` / ``max`` / ``step``; ``select`` carries
``options``. ``random_state`` is not a model parameter — it is controlled
by the experiment-level ``random_state`` field.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backend.services.ml_contracts import ModelAdapter


class _SklearnModelAdapter:
    """Generic adapter wrapping a sklearn-style estimator class.

    ``default_parameters`` is derived from ``parameters`` defaults so it
    always matches the metadata the frontend renders.
    """

    def __init__(
        self,
        name: str,
        cls: type,
        task_type: str,
        description: str,
        parameters: Sequence[dict[str, Any]],
    ) -> None:
        self.name = name
        self._cls = cls
        self.task_type = task_type
        self.description = description
        self.parameters = list(parameters)
        self.default_parameters = {p["name"]: p["default"] for p in self.parameters}

    def create_estimator(
        self,
        hyperparameters: dict[str, Any] | None = None,
        random_state: int | None = None,
    ) -> Any:
        params = {**self.default_parameters, **(hyperparameters or {})}
        if random_state is not None and _accepts_random_state(self._cls):
            params["random_state"] = random_state
        return self._cls(**params)


# --- registry ---------------------------------------------------------------


def _accepts_random_state(cls: type) -> bool:
    """Whether ``cls.__init__`` exposes a ``random_state`` parameter.

    Lets the pipeline inject the experiment-level seed into models that
    support it (tree ensembles) without breaking models that do not
    (KNN/NB, the GD regressors).
    """
    import inspect

    return "random_state" in inspect.signature(cls.__init__).parameters


def _build_registry() -> dict[str, ModelAdapter]:
    """Build the registry from models already merged into develop.

    Every import below is explicit and must succeed. Add a new model by
    importing its class and appending an entry — do not wrap imports in
    try/except, so that integration errors surface immediately.
    """
    registry: dict[str, ModelAdapter] = {}

    # --- A: KNN & Gaussian Naive Bayes (merged via PR #2) -------------------
    # A exposes a dict-shaped ``parameters_metadata`` class attribute; the
    # list form below is maintained here so the registry stays the single
    # source of truth for frontend form metadata. Values match A's defaults.
    from ml_engine.models.knn import KNNModel
    from ml_engine.models.naive_bayes import GaussianNBModel

    registry["knn_classifier"] = _SklearnModelAdapter(
        name="knn_classifier",
        cls=KNNModel,
        task_type="classification",
        description="K 近邻分类器。",
        parameters=[
            {"name": "n_neighbors", "label": "邻居数", "type": "int",
             "default": 5, "description": "选择最近邻居的数量。",
             "min": 1, "max": 20, "step": 1},
            {"name": "weights", "label": "权重方式", "type": "select",
             "default": "uniform", "description": "预测时邻居的权重方式。",
             "options": ["uniform", "distance"]},
            {"name": "p", "label": "距离度量", "type": "int",
             "default": 2, "description": "1 为曼哈顿距离，2 为欧氏距离。",
             "min": 1, "max": 2, "step": 1},
        ],
    )
    registry["gaussian_nb"] = _SklearnModelAdapter(
        name="gaussian_nb",
        cls=GaussianNBModel,
        task_type="classification",
        description="高斯朴素贝叶斯分类器。",
        parameters=[
            {"name": "var_smoothing", "label": "方差平滑", "type": "float",
             "default": 1e-9, "description": "防止方差为零的平滑参数。",
             "min": 1e-12, "max": 1e-1, "step": 1e-10},
        ],
    )

    # --- B: linear & ridge regression (merged via PR #3) --------------------
    # B's estimators are plain numpy GD classes with no metadata attribute,
    # so the form specs live here. learning_rate must be strictly > 0 (its
    # min is 1e-4, never 0). max_iter is capped at 10000 in the form because
    # every loss value is kept in ``loss_history_`` and stored in the DB.
    from ml_engine.models import LinearRegressionGD, RidgeRegressionGD

    gd_parameters = [
        {
            "name": "learning_rate",
            "label": "学习率",
            "type": "float",
            "default": 0.05,
            "description": "梯度下降每一步的步长，必须严格大于 0。",
            "min": 0.0001,
            "max": 1.0,
            "step": 0.01,
        },
        {
            "name": "max_iter",
            "label": "最大迭代次数",
            "type": "int",
            "default": 1000,
            "description": "达到此次数后即使未收敛也停止。",
            "min": 1,
            "max": 10000,
        },
        {
            "name": "tol",
            "label": "收敛容差",
            "type": "float",
            "default": 1e-8,
            "description": "损失变化小于该值时视为收敛。",
            "min": 0.0,
            "max": 1.0,
        },
    ]
    registry["linear_regression"] = _SklearnModelAdapter(
        name="linear_regression",
        cls=LinearRegressionGD,
        task_type="regression",
        description="基于梯度下降的线性回归。",
        parameters=gd_parameters,
    )
    registry["ridge_regression"] = _SklearnModelAdapter(
        name="ridge_regression",
        cls=RidgeRegressionGD,
        task_type="regression",
        description="岭回归（L2 正则化线性回归）。",
        parameters=[
            *gd_parameters,
            {
                "name": "alpha",
                "label": "正则化强度",
                "type": "float",
                "default": 1.0,
                "description": "L2 惩罚项系数，越大正则化越强。",
                "min": 0.0,
                "max": 100.0,
            },
        ],
    )

    # --- B: classification models (merged via PR #8 / #10 / #11) ------------
    # LogisticRegressionGD (PR #8), LinearSVM (PR #10), MLPClassifier (PR #11).
    # The logistic regressor shares ``gd_parameters`` (identical defaults to
    # linear/ridge); LinearSVM and MLPClassifier default to learning_rate
    # 0.01, so their form specs are defined separately. ``random_state`` is
    # never listed in the form: the adapter injects the experiment-level
    # seed for models that accept it (MLPClassifier).
    from ml_engine.models import LinearSVM, LogisticRegressionGD, MLPClassifier

    registry["logistic_regression"] = _SklearnModelAdapter(
        name="logistic_regression",
        cls=LogisticRegressionGD,
        task_type="classification",
        description="基于 Softmax 的多分类逻辑回归（全批量梯度下降）。",
        parameters=[
            *gd_parameters,
            {
                "name": "alpha",
                "label": "正则化强度",
                "type": "float",
                "default": 0.0,
                "description": "L2 惩罚项系数，0 表示不正则化，越大正则化越强。",
                "min": 0.0,
                "max": 100.0,
            },
        ],
    )
    registry["linear_svm"] = _SklearnModelAdapter(
        name="linear_svm",
        cls=LinearSVM,
        task_type="classification",
        description="多分类线性支持向量机（One-vs-Rest + Hinge Loss，全批量次梯度下降）。",
        parameters=[
            {
                "name": "learning_rate",
                "label": "学习率",
                "type": "float",
                "default": 0.01,
                "description": "次梯度下降每一步的步长，必须严格大于 0。",
                "min": 0.0001,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "max_iter",
                "label": "最大迭代次数",
                "type": "int",
                "default": 1000,
                "description": "达到此次数后即使未收敛也停止。",
                "min": 1,
                "max": 10000,
            },
            {
                "name": "tol",
                "label": "收敛容差",
                "type": "float",
                "default": 1e-8,
                "description": "目标函数值变化小于该值时视为收敛。",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "C",
                "label": "正则化参数 C",
                "type": "float",
                "default": 1.0,
                "description": "Hinge Loss 的正则化参数，越大正则化越弱。",
                "min": 0.001,
                "max": 1000.0,
            },
        ],
    )
    registry["mlp_classifier"] = _SklearnModelAdapter(
        name="mlp_classifier",
        cls=MLPClassifier,
        task_type="classification",
        description="单隐藏层多层感知机（Linear-ReLU-Linear-Softmax，全批量梯度下降）。",
        parameters=[
            {
                "name": "hidden_size",
                "label": "隐藏层神经元数",
                "type": "int",
                "default": 32,
                "description": "隐藏层神经元数量，越大模型容量越强、训练越慢。",
                "min": 1,
                "max": 256,
                "step": 1,
            },
            {
                "name": "learning_rate",
                "label": "学习率",
                "type": "float",
                "default": 0.01,
                "description": "梯度下降每一步的步长，必须严格大于 0。",
                "min": 0.0001,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "max_iter",
                "label": "最大迭代次数",
                "type": "int",
                "default": 1000,
                "description": "达到此次数后即使未收敛也停止。",
                "min": 1,
                "max": 10000,
            },
            {
                "name": "tol",
                "label": "收敛容差",
                "type": "float",
                "default": 1e-8,
                "description": "损失变化小于该值时视为收敛。",
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "alpha",
                "label": "正则化强度",
                "type": "float",
                "default": 0.0001,
                "description": "L2 惩罚项系数（仅作用于权重矩阵，0 表示不正则化）。",
                "min": 0.0,
                "max": 100.0,
            },
        ],
    )

    # --- C: regression trees (merged via feature/tree_ensemble) -------------
    # Form specs below mirror C's ``model_params_meta.json``. ``random_state``
    # is stripped: it is controlled by the experiment-level field. The null
    # select option of max_features is dropped because ParameterSpec.options
    # is a plain string list (None stays supported through the direct API).
    from ml_engine import (
        CartRegressionTree,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )

    registry["decision_tree_regressor"] = _SklearnModelAdapter(
        name="decision_tree_regressor",
        cls=CartRegressionTree,
        task_type="regression",
        description="单棵 CART 回归树，按 MSE 增益递归二分，叶子取节点内标签均值。",
        parameters=[
            {
                "name": "max_depth",
                "label": "最大深度",
                "type": "int",
                "default": None,
                "description": "树的最大深度；留空表示不限制，树将持续生长。"
                "深度越大越容易过拟合。",
                "min": 1,
                "max": 32,
                "step": 1,
            },
            {
                "name": "min_samples_split",
                "label": "节点最小分裂样本数",
                "type": "int",
                "default": 2,
                "description": "一个节点至少包含多少个样本才允许继续分裂，用于控制过拟合。",
                "min": 2,
                "max": 64,
                "step": 1,
            },
        ],
    )
    registry["random_forest_regressor"] = _SklearnModelAdapter(
        name="random_forest_regressor",
        cls=RandomForestRegressor,
        task_type="regression",
        description="多棵随机特征 CART 树的 Bagging 集成：bootstrap 抽样训练，预测取平均。",
        parameters=[
            {
                "name": "n_estimators",
                "label": "树的数量",
                "type": "int",
                "default": 100,
                "description": "森林中基学习器（CART 树）的数量；越多越稳定但训练越慢。",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            {
                "name": "max_depth",
                "label": "单棵树最大深度",
                "type": "int",
                "default": None,
                "description": "单棵树的最大深度；留空表示不限制。",
                "min": 1,
                "max": 32,
                "step": 1,
            },
            {
                "name": "max_features",
                "label": "分裂候选特征数",
                "type": "select",
                "default": "sqrt",
                "description": "每次节点分裂时随机采样参与搜索的特征数策略。",
                "options": ["sqrt", "log2"],
            },
        ],
    )
    registry["gbdt_regressor"] = _SklearnModelAdapter(
        name="gbdt_regressor",
        cls=GradientBoostingRegressor,
        task_type="regression",
        description="梯度提升回归：每棵 CART 树拟合当前残差，按学习率缩减累加。",
        parameters=[
            {
                "name": "n_estimators",
                "label": "提升轮数",
                "type": "int",
                "default": 100,
                "description": "提升迭代轮数，即残差树的数量；轮数越多训练误差越低。",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            {
                "name": "learning_rate",
                "label": "学习率",
                "type": "float",
                "default": 0.1,
                "description": "每棵残差树贡献的缩放步长，必须大于 0。",
                "min": 0.001,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "max_depth",
                "label": "单棵树最大深度",
                "type": "int",
                "default": 3,
                "description": "个体残差回归树的最大深度；GBDT 通常用浅树（默认 3）。",
                "min": 1,
                "max": 32,
                "step": 1,
            },
        ],
    )

    return registry


_MODELS: dict[str, ModelAdapter] | None = None


def _registry() -> dict[str, ModelAdapter]:
    global _MODELS
    if _MODELS is None:
        _MODELS = _build_registry()
    return _MODELS


def get_model(name: str) -> ModelAdapter:
    registry = _registry()
    if name not in registry:
        raise KeyError(name)
    return registry[name]


def list_models(task_type: str | None = None) -> list[ModelAdapter]:
    registry = _registry()
    if task_type is None:
        return list(registry.values())
    return [m for m in registry.values() if m.task_type == task_type]
