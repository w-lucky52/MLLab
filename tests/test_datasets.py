# tests/test_a_datasets.py
from ml_engine.datasets import get_dataset, split_dataset
from ml_engine.datasets.utils import parse_csv
from ml_engine.evaluation.classification import evaluate
from ml_engine.models.knn import KNNModel
from ml_engine.models.naive_bayes import GaussianNBModel


def test_classification_iris():
    """测试 iris 数据集的加载、切分、模型预测与评估全链路"""
    data = get_dataset("iris").load()
    assert data.task_type == "classification"
    assert data.X.shape == (150, 4)

    # 注意新契约：X 和 y 拆开传，返回 4 元组
    X_train, X_test, y_train, y_test = split_dataset(
        data.X, data.y, test_size=0.3, task_type=data.task_type
    )

    # 测试 KNN
    knn = KNNModel().fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    result_knn = evaluate(y_test, y_pred_knn, class_labels=data.target_names)
    
    # 验证指标正确性和类别标签（防止返回数字标签）
    assert result_knn["metrics"]["accuracy"] > 0.8
    assert "setosa" in result_knn["class_labels"]

    # 测试高斯朴素贝叶斯
    nb = GaussianNBModel().fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    result_nb = evaluate(y_test, y_pred_nb, class_labels=data.target_names)
    assert result_nb["metrics"]["accuracy"] > 0.8


def test_classification_digits():
    """测试 digits 数据集（64个特征，10分类）能被正确处理"""
    data = get_dataset("digits").load()
    assert data.X.shape == (1797, 64)
    assert len(data.target_names) == 10

    X_train, X_test, y_train, y_test = split_dataset(
        data.X, data.y, test_size=0.3, task_type=data.task_type
    )
    # 确认切分后的维度正确
    assert X_train.shape[1] == 64
    assert X_test.shape[1] == 64


def test_regression_diabetes_data():
    """测试 diabetes 数据集（回归任务）的加载与切分"""
    data = get_dataset("diabetes").load()
    assert data.task_type == "regression"
    assert data.X.shape == (442, 10)

    X_train, X_test, y_train, y_test = split_dataset(
        data.X, data.y, test_size=0.3, task_type=data.task_type
    )
    # 验证切分后的样本数（442 * 0.7 = 309.4 => 309）
    assert X_train.shape[0] == 309
    assert X_test.shape[0] == 133


def test_parse_csv(tmp_path):
    """测试 CSV 解析函数（利用 pytest 的临时目录）"""
    # 创建一个临时测试 CSV 文件
    csv_file = tmp_path / "test_dummy.csv"
    csv_content = "feature1,feature2,target\n1.0,2.0,0\n3.0,,1\n5.0,6.0,0"
    csv_file.write_text(csv_content, encoding="utf-8")

    result = parse_csv(str(csv_file))

    assert result["columns"] == ["feature1", "feature2", "target"]
    assert result["target_candidates"] == ["target"]
    assert result["missing_counts"]["feature2"] == 1
    assert len(result["preview"]) == 3