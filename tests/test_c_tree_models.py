from ml_engine import CartRegressionTree, GradientBoostingRegressor, RandomForestRegressor
from ml_engine.datasets import get_dataset, split_dataset
from ml_engine.evaluation.regression import evaluate as eval_regression


def test_tree_models_on_diabetes():
    """用 diabetes 数据集验证 C 的三个树模型全链路"""
    data = get_dataset("diabetes").load()
    assert data.task_type == "regression"

    X_train, X_test, y_train, y_test = split_dataset(
        data.X, data.y, test_size=0.3, task_type=data.task_type
    )

    # 使用 D 提供的正确类名导入
    models = {
        "decision_tree_regressor": CartRegressionTree(),
        "random_forest_regressor": RandomForestRegressor(),
        "gbdt_regressor": GradientBoostingRegressor(),
    }

    for model_id, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = eval_regression(y_test, y_pred)

        # 验证回归指标键名是否和全组约定一致
        assert "mae" in result["metrics"]
        assert "mse" in result["metrics"]
        assert "rmse" in result["metrics"]
        assert "r2" in result["metrics"]
        
        # 去掉了 print，完全符合 D 的要求

# 注意：文件最末尾一定要留一个空行（按下回车即可），防止 ruff 挂 W292 错误