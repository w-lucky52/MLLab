# tests/test_c_tree_models.py
import pytest
from ml_engine.datasets import get_dataset, split_dataset
from ml_engine.evaluation.regression import evaluate as eval_regression


def test_tree_models_on_diabetes():
    """用 diabetes 数据集验证 C 的三个树模型全链路"""
    data = get_dataset("diabetes").load()
    assert data.task_type == "regression"

    X_train, X_test, y_train, y_test = split_dataset(
        data.X, data.y, test_size=0.3, task_type=data.task_type
    )

    # 按 C 提交的注册 ID 导入模型
    from ml_engine.models.decision_tree import DecisionTreeRegressorModel
    from ml_engine.models.random_forest import RandomForestRegressorModel
    from ml_engine.models.gbdt import GBDTRegressorModel

    models = {
        "decision_tree_regressor": DecisionTreeRegressorModel(),
        "random_forest_regressor": RandomForestRegressorModel(),
        "gbdt_regressor": GBDTRegressorModel(),
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

        print(f"{model_id}: {result['metrics']}")