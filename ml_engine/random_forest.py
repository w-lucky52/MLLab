# -*- coding: utf-8 -*-
"""
随机森林回归（学生C / 树模型体系，P1）

实现思路：
1. 基学习器 100% 复用 ml_engine.cart.CartRegressionTree：
   _RandomTree 是它的子类，只覆写 _find_best_split —— 先随机抽取 max_features
   个特征列，再调用父类的 _find_best_split 完成“遍历阈值 + 计算MSE增益”的全部
   核心计算，最后把局部列号映射回原始特征号。树构建 _build_tree、切分 _split、
   递归预测 predict、特征重要性统计等逻辑全部继承自 CartRegressionTree，不重写。
2. RandomForestRegressor 负责：bootstrap 有放回抽样训练多棵随机化树、
   预测结果取算术平均、特征重要性取各树平均后归一化。
"""
import numpy as np

from ml_engine.cart import CartRegressionTree


class _RandomTree(CartRegressionTree):
    """带“分裂时随机特征采样”的CART回归树（随机森林内部基学习器）。

    除 _find_best_split 外，所有行为与 CartRegressionTree 完全一致；
    因此 isinstance(_RandomTree(...), CartRegressionTree) 为 True。
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, rng=None):
        """
        max_features：整数，每次分裂允许参与搜索的特征数（由森林层按总特征数解析后传入）
        rng：numpy.random.Generator，由森林层统一创建，保证整体可复现
        """
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        self.max_features = max_features
        #调用方未传时自建随机数生成器，保证单棵树单独使用时也能运行
        self.rng = rng if rng is not None else np.random.default_rng()

    def _find_best_split(self, X, y):
        """只在随机采样的 max_features 个特征中寻找最优分裂。

        具体做法：抽出特征子集 X_sub 后，直接调用父类
        CartRegressionTree._find_best_split（MSE增益、阈值遍历等核心逻辑复用），
        再把返回的“子集局部列号”映射回“原始特征号”。
        """
        n_features = X.shape[1]
        k = min(self.max_features, n_features)
        #全部特征都参与时直接复用父类原逻辑：不做列置换，
        #保证并列增益下的平局选择与原生 CartRegressionTree 完全一致
        if k >= n_features:
            return super()._find_best_split(X, y)
        #无放回随机抽取 k 个候选特征
        chosen = self.rng.choice(n_features, size=k, replace=False)
        X_sub = X[:, chosen]
        feat_sub, thresh, gain = super()._find_best_split(X_sub, y)
        #兜底：随机子集在当前节点无有效分裂时，用全部特征再试一次，
        #避免因抽到的特征在该节点都为常数而提前停止（与sklearn处理思想一致）
        if feat_sub is None and k < n_features:
            feat_full, thresh, gain = super()._find_best_split(X, y)
            return feat_full, thresh, gain
        if feat_sub is None:
            return None, thresh, gain
        #局部列号 -> 原始特征号；阈值与增益数值不变
        return chosen[feat_sub], thresh, gain


class RandomForestRegressor:
    """随机森林回归器，接口风格对齐 sklearn：
        fit(X, y) 训练；predict(X) 预测；feature_importances_ 特征重要性。
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 max_features="sqrt", random_state=None):
        """
        n_estimators：森林中CART树的数量，默认100
        max_depth：单棵树最大深度，默认None（不限制）
        max_features：每次分裂随机采样的特征数量，支持：
            int   —— 直接指定特征数
            float —— 按总特征数比例（取值范围 (0, 1]）
            "sqrt"—— sqrt(n_features)（默认）
            "log2"—— log2(n_features)
            None  —— 使用全部特征
        random_state：随机种子（int），保证bootstrap抽样与特征采样可复现
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        #训练产物：所有基学习器、每棵树的bootstrap样本下标、特征重要性
        self.estimators_ = []
        self.estimators_samples_ = []
        self.feature_importances_ = None

    def _resolve_max_features(self, n_features):
        """把构造参数 max_features 解析成“每次分裂参与搜索的特征数(>=1)”。"""
        mf = self.max_features
        if mf is None:
            k = n_features
        elif isinstance(mf, bool):
            #bool 是 int 的子类，显式拒绝以免被误当成 0/1
            raise ValueError("max_features 不支持布尔值")
        elif isinstance(mf, int):
            if mf < 1:
                raise ValueError("max_features 为整数时必须 >= 1")
            k = mf
        elif isinstance(mf, float):
            if not 0.0 < mf <= 1.0:
                raise ValueError("max_features 为浮点数时必须在 (0, 1] 范围内")
            #与sklearn一致：按比例向下取整，至少保留1个特征
            k = int(n_features * mf)
        elif mf == "sqrt":
            k = int(np.sqrt(n_features))
        elif mf == "log2":
            k = int(np.log2(n_features))
        else:
            raise ValueError(
                "max_features 仅支持 int、(0,1] 内的 float、'sqrt'、'log2' 或 None，"
                "当前收到：{!r}".format(mf)
            )
        #裁剪到 [1, n_features]
        return max(1, min(k, n_features))

    def fit(self, X, y):
        """训练随机森林。

        流程（对每棵树重复 n_estimators 次）：
        1) bootstrap 有放回抽样：从 n 个样本中等概率抽 n 个（允许重复/遗漏），
           袋外样本天然构成 OOB；
        2) 在 bootstrap 数据上训练一棵 _RandomTree，每次分裂只在随机特征子集中搜索；
        3) 保存树对象与本次抽样下标。
        最后：各树 feature_importances_ 取算术平均并归一化。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 的样本数不一致")

        n_samples, n_features = X.shape
        k_features = self._resolve_max_features(n_features)
        #全森林共用一个随机数流，random_state 固定则抽样顺序完全可复现
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        self.estimators_samples_ = []
        for _ in range(self.n_estimators):
            #1) bootstrap：有放回抽取与原样本等量的下标
            sample_idx = rng.integers(0, n_samples, size=n_samples)
            X_boot, y_boot = X[sample_idx], y[sample_idx]
            #2) 训练随机特征CART（分裂/预测逻辑全部继承自 CartRegressionTree）
            tree = _RandomTree(
                max_depth=self.max_depth,
                max_features=k_features,
                rng=rng,
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            self.estimators_samples_.append(sample_idx)

        #3) 森林特征重要性 = 各树归一化重要性的算术平均，再整体归一化使总和为1
        imp_sum = np.zeros(n_features, dtype=float)
        for tree in self.estimators_:
            imp_sum += tree.feature_importances_
        imp_mean = imp_sum / self.n_estimators
        total = imp_mean.sum()
        if total > 0:
            self.feature_importances_ = imp_mean / total
        else:
            #边界：所有树都没有产生任何分裂时，重要性全为0
            self.feature_importances_ = imp_mean
        return self

    def predict(self, X):
        """集成预测：每棵树独立预测，最终结果取算术平均（回归用平均降低方差）。"""
        if not self.estimators_:
            raise ValueError("模型尚未训练，请先调用 fit(X, y)")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        #所有树的预测堆叠成 (n_estimators, n_samples)，沿树轴取均值
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(all_preds, axis=0)
