# -*- coding: utf-8 -*-
"""
GBDT 梯度提升回归（学生C / 树模型体系，P2）

算法原理（平方损失）：
    损失 L = 1/2 * (y - F(x))^2，对当前预测 F 的负梯度恰好等于残差 r = y - F。
    因此 GBDT 回归每一轮用一棵 CART 回归树去拟合当前残差，再沿负梯度方向
    （即残差方向）以 learning_rate 为步长更新预测，逐步减小训练误差：
        F_0(x)   = mean(y)                              # 初始基准：标签均值
        r_m      = y - F_{m-1}(X)                      # 本轮负梯度=残差
        h_m      = CART.fit(X, r_m)                    # 基学习器拟合残差
        F_m(X)   = F_{m-1}(X) + learning_rate * h_m(X) # 缩减更新
    预测：F(x) = F_0 + learning_rate * Σ h_m(x)

实现约束：
    基学习器 100% 复用 ml_engine.cart.CartRegressionTree（直接实例化使用），
    树的分裂、构建、预测、单树特征重要性逻辑全部不重写。
"""
import numpy as np

from ml_engine.cart import CartRegressionTree


class GradientBoostingRegressor:
    """GBDT回归器，接口风格对齐 sklearn：
        fit(X, y) 训练；predict(X) 预测；feature_importances_ 特征重要性。
    """

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=None):
        """
        n_estimators：提升轮数（基学习器数量），默认100
        learning_rate：学习率/缩减系数，每棵树的贡献按此比例缩小，默认0.1
        max_depth：单棵残差回归树的最大深度，默认3（弱学习器，浅树+小步长）
        random_state：随机种子，仅为与sklearn接口对齐而保留；
                      平方损失+本项目CART的训练过程本身是确定性的，不使用随机性
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        #训练产物：初始基准值、按轮保存的基学习器、特征重要性、训练MSE曲线
        self.init_value_ = None
        self.estimators_ = []
        self.feature_importances_ = None
        self.train_score_ = None

    def fit(self, X, y):
        """训练GBDT（前向分步加法模型）。

        每轮：算残差 r=y-F -> 用 CartRegressionTree 拟合 r ->
              F += learning_rate * 树预测，并保存该树。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 的样本数不一致")
        if self.n_estimators < 1:
            raise ValueError("n_estimators 必须 >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须 > 0")

        n_samples, n_features = X.shape

        #1) 初始化：F_0 = 训练标签均值（平方损失下的最优常数预测）
        self.init_value_ = float(np.mean(y))
        current_pred = np.full(n_samples, self.init_value_, dtype=float)

        self.estimators_ = []
        self.train_score_ = np.empty(self.n_estimators, dtype=float)
        #累加各树“归一化后的特征重要性”，最后取平均再整体归一化
        imp_sum = np.zeros(n_features, dtype=float)

        #2) 前向分步迭代
        for m in range(self.n_estimators):
            #2.1 负梯度（平方损失下即残差）：r = y - F
            residual = y - current_pred
            #2.2 训练一棵CART回归树拟合残差（分裂/预测逻辑100%复用现有CART）
            tree = CartRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residual)
            #2.3 沿残差方向以 learning_rate 为步长更新预测值
            update = tree.predict(X)
            current_pred = current_pred + self.learning_rate * update
            #2.4 保存基学习器、记录本轮训练MSE、累加该树特征重要性
            self.estimators_.append(tree)
            imp_sum += tree.feature_importances_
            self.train_score_[m] = float(np.mean((y - current_pred) ** 2))

        #3) 模型特征重要性：各树归一化重要性取算术平均，再整体归一化使总和为1
        imp_mean = imp_sum / self.n_estimators
        total = imp_mean.sum()
        if total > 0:
            self.feature_importances_ = imp_mean / total
        else:
            #边界：所有轮次残差都无有效分裂（如标签为常数），重要性全为0
            self.feature_importances_ = imp_mean
        return self

    def predict(self, X):
        """集成预测：初始基准值 + learning_rate × 所有残差树预测的累加和。"""
        if not self.estimators_:
            raise ValueError("模型尚未训练，请先调用 fit(X, y)")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        #从常数基准出发，逐棵累加缩减后的树预测
        pred = np.full(X.shape[0], self.init_value_, dtype=float)
        for tree in self.estimators_:
            pred += self.learning_rate * tree.predict(X)
        return pred
