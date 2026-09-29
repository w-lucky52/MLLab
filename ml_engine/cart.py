import numpy as np

class CartRegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        max_depth：树最大深度，防止树长得太深过拟合
        min_samples_split：一个节点至少要有多少样本，才允许继续分裂
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None        # 用来存整棵树结构
        self.feature_importances_ = None # 特征重要性，后面要输出给前端E同学

    def _mse(self,y):
        """
        y：一组标签值。MSE均方误差，数值越大代表这一组数据波动越大。
        CART找分裂点：分裂之后左右两边整体MSE越小越好
        """
        return np.mean((y - np.mean(y)) ** 2)

    def _split(self, X, y, feat_idx, threshold):
        """
        X：特征矩阵；y：标签；
        feat_idx：第几号特征；threshold：分割阈值
        规则：小于等于阈值去左边，剩下去右边
        返回：左数据集、左标签、右数据集、右标签
        """
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _find_best_split(self, X, y):
        """遍历全部特征、全部阈值，找收益最大的分裂点"""
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feat = None
        best_thresh = None
        base_mse = self._mse(y)

        for feat_idx in range(n_features):
            #取出这个特征所有不重复的值作为候选分割阈值
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                X_left, y_left, X_right, y_right = self._split(X,y,feat_idx,thresh)
                #如果切完某一边没有样本，跳过这个分割
                if len(y_left)==0 or len(y_right)==0:
                    continue
                mse_left = self._mse(y_left)
                mse_right = self._mse(y_right)
                #计算分裂带来误差下降，叫增益gain
                gain = base_mse - (len(y_left)/n_samples*mse_left + len(y_right)/n_samples*mse_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh
        return best_feat, best_thresh, best_gain

    def _build_tree(self, X, y, depth):
        """
        递归构建树
        """
        n_samples = X.shape[0]
        #停止递归条件：达到最大深度 / 样本数量不足，就生成叶子节点
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return {"leaf":True, "value":np.mean(y)}
        feat, thresh, gain = self._find_best_split(X,y)
        #找不到任何有效分裂，直接叶子节点
        if feat is None:
            return {"leaf":True, "value":np.mean(y)}
        Xl,yl,Xr,yr = self._split(X,y,feat,thresh)
        left_node = self._build_tree(Xl,yl,depth+1)
        right_node = self._build_tree(Xr,yr,depth+1)
        #非叶子节点，保存分裂信息
        return {"leaf":False,"feature":feat,"threshold":thresh,"left":left_node,"right":right_node,"gain":gain}

    def _accumulate_gain(self, node, total_gain):
        """
        （新增功能）递归遍历已构建好的树，累加每个分裂节点的MSE增益。
        node：当前树节点；total_gain：长度等于特征数的累加数组（原地累加）
        说明：非叶子节点中的 gain 直接复用构建时 _find_best_split 的返回值，
              不重新计算增益；叶子节点不产生增益。
        """
        if node["leaf"]:
            return
        #本次分裂使用的特征，累加本次分裂的MSE增益
        total_gain[node["feature"]] += node["gain"]
        self._accumulate_gain(node["left"], total_gain)
        self._accumulate_gain(node["right"], total_gain)

    def fit(self, X, y):
        """
        对外训练接口：输入特征X，标签y，完成模型训练
        """
        self.tree = self._build_tree(X,y,depth=0)
        # ===== 特征重要性统计（新增功能，不影响上方树构建逻辑）=====
        n_features = X.shape[1]
        #1) 各特征总增益先置0
        total_gain = np.zeros(n_features, dtype=float)
        #2) 遍历整棵树，按分裂特征累加各节点 gain（gain 直接来自 _find_best_split）
        self._accumulate_gain(self.tree, total_gain)
        #3) 归一化：各特征总增益 ÷ 全部分裂总增益，使重要性之和为1
        gain_sum = total_gain.sum()
        if gain_sum > 0:
            self.feature_importances_ = total_gain / gain_sum
        else:
            #边界：整棵树无任何有效分裂（如单一样本/标签无波动），重要性全为0
            self.feature_importances_ = total_gain
        return self

    def _predict_one(self,x,node):
        """对一条样本做预测，递归往下走到叶子拿结果"""
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])
    def predict(self,X):
        """对外批量预测接口,D同学调用这个得到预测结果"""
        res = []
        for row in X:
            res.append(self._predict_one(row, self.tree))
        return np.array(res)