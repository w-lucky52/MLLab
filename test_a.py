from ml_engine.datasets.classification import IrisAdapter
from ml_engine.datasets.base import split_dataset
from ml_engine.models.knn import KNNModel
from ml_engine.models.naive_bayes import GaussianNBModel
from sklearn.metrics import accuracy_score

data = IrisAdapter().load()
print("task_type:", data.task_type)
print("X shape:", data.X.shape, "y shape:", data.y.shape)

X_train, X_test, y_train, y_test, scaler = split_dataset(data, test_size=0.3)

for Model in [KNNModel, GaussianNBModel]:
    model = Model().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{Model.__name__} accuracy: {accuracy_score(y_test, y_pred):.4f}")

from ml_engine.datasets.regression import DiabetesAdapter
# 注意：这里不需要引入任何模型类，只测数据部分

print("\n--- 验证 Diabetes (回归) 数据集 ---")
# 1. 加载数据
data_reg = DiabetesAdapter().load()
print("task_type:", data_reg.task_type)
print("X shape:", data_reg.X.shape, "y shape:", data_reg.y.shape)
print("feature_names 数量:", len(data_reg.feature_names) if data_reg.feature_names else 0)

# 2. 测试切分与防泄漏逻辑
X_train_r, X_test_r, y_train_r, y_test_r, scaler_r = split_dataset(data_reg, test_size=0.3)
print("切分完成 -> X_train:", X_train_r.shape, "X_test:", X_test_r.shape)

# 3. 验证是否在训练集上 fit 了 Scaler
print("scaler 是否已创建 (在训练集上fit):", scaler_r is not None)


from ml_engine.datasets.utils import parse_csv
import os

print("\n--- 验证 parse_csv 函数 ---")

# 1. 自动创建一个测试用的假 CSV 文件
test_csv_path = "test_dummy.csv"
with open(test_csv_path, "w", encoding="utf-8") as f:
    f.write("feature1,feature2,target\n1.0,2.0,0\n3.0,,1\n5.0,6.0,0\n7.0,8.0,1\n9.0,10.0,0")

# 2. 调用你写的函数
try:
    result = parse_csv(test_csv_path)
    
    # 3. 打印输出，检查格式对不对
    print("✅ 解析成功！")
    print("识别到的列名 (columns):", result["columns"])
    print("推荐的目标列 (target_candidates):", result["target_candidates"])
    print("缺失值统计 (missing_counts):", result["missing_counts"])
    print("预览数据行数 (preview):", len(result["preview"]))
    
except Exception as e:
    print("❌ 解析报错：", e)

# 4. 清理临时文件
if os.path.exists(test_csv_path):
    os.remove(test_csv_path)