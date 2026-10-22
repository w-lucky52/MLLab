# MLLab

MLLab 是一个可扩展的机器学习实验平台：选择内置数据集与算法模型，配置超参数后提交实验，异步训练并可视化评估结果。

项目按 **系统层 / 算法层** 分离：`backend` 负责 HTTP 接口、数据持久化与任务编排，`ml_engine` 只依赖 NumPy / scikit-learn，不感知 Web 框架。双方通过 `backend/services/ml_contracts.py` 中的 Protocol 接口通信，新增数据集或模型无需改动 API。

## 功能特性

- **数据集浏览**：内置 5 个数据集，提供样本数、特征数与预览表格
- **模型目录**：内置 10 个模型，附带超参数元信息（类型、默认值、取值区间），前端据此动态生成参数表单
- **实验执行**：提交后异步运行，状态按 `pending → running → completed / failed` 流转
- **结果可视化**：分类展示指标卡片与混淆矩阵，回归展示指标卡片与「真实值 vs 预测值」散点图
- **实验管理**：历史列表、详情查看、删除（同时清理产物目录）

## 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | FastAPI、Pydantic v2、SQLAlchemy 2、SQLite、Uvicorn |
| 算法层 | NumPy、scikit-learn、pandas、matplotlib |
| 前端 | Vue 3、Vite、Vue Router、Element Plus、ECharts |

## 环境要求

- Python 3.10+（推荐 3.11）
- Node.js `^22.18.0` 或 `>=24.12.0`
- npm

## 快速开始

### 后端

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[ml,dev]"
uvicorn backend.main:app --reload --port 8000
```

启动后：

- 服务首页：<http://127.0.0.1:8000/>
- 健康检查：<http://127.0.0.1:8000/api/v1/health>
- 接口文档：<http://127.0.0.1:8000/docs>

如需自定义配置，可复制配置模板（所有变量均以 `MLLAB_` 为前缀）：

```bash
cp .env.example .env
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

打开 <http://localhost:5173/>。Vite 已配置 `/api` 代理到 `http://127.0.0.1:8000`，本地开发无需额外处理跨域。

> 前后端需同时运行；前端全部数据来自后端接口，没有 mock 数据。

## 目录结构

```text
MLLab/
├── backend/
│   ├── api/
│   │   ├── router.py            # 版本化路由聚合
│   │   ├── schemas.py           # 请求/响应模型（对外契约）
│   │   └── routes/              # health / datasets / models / experiments
│   ├── core/config.py           # 配置（环境变量 + 默认值）
│   ├── database/                # SQLAlchemy 模型与会话
│   ├── services/
│   │   ├── experiment_service.py # 实验创建、执行、查询、删除
│   │   ├── ml_bridge.py         # 适配 ml_engine，注入随机种子
│   │   └── ml_contracts.py      # 与算法层的 Protocol 契约
│   └── main.py                  # 应用入口、CORS、产物静态目录
├── ml_engine/
│   ├── datasets/                # 数据集注册、切分与预览
│   ├── models/                  # KNN、朴素贝叶斯、逻辑回归、SVM、MLP、线性/岭回归
│   ├── cart.py                  # CART 回归树
│   ├── random_forest.py         # 随机森林回归
│   ├── gbdt.py                  # 梯度提升回归树
│   ├── evaluation/              # 统一评估入口（按任务类型分发）
│   └── metrics/                 # 回归指标实现
├── frontend/                    # Vue 3 单页应用
├── tests/                       # pytest 用例
├── experiments/                 # 实验产物（图表、summary.json），不提交 Git
├── model_params_meta.json       # 模型超参数元信息
├── .env.example
└── pyproject.toml
```

## API 概览

所有业务接口挂载在 `/api/v1` 前缀下，完整定义以 `/docs` 为准。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/datasets` | 数据集列表 |
| GET | `/api/v1/datasets/{name}` | 数据集详情（含预览行） |
| GET | `/api/v1/models` | 模型列表，支持 `?task_type=classification\|regression` 过滤 |
| GET | `/api/v1/models/{name}` | 模型详情（含超参数元信息） |
| POST | `/api/v1/experiments` | 创建实验，返回 201，后台异步执行 |
| GET | `/api/v1/experiments` | 实验列表（按 id 倒序） |
| GET | `/api/v1/experiments/{id}` | 实验详情，前端轮询该接口获取状态 |
| DELETE | `/api/v1/experiments/{id}` | 删除实验及其产物 |
| GET | `/artifacts/{artifact_dir}/{file}` | 图表等实验产物静态文件 |

创建实验示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "iris",
    "model_name": "knn_classifier",
    "test_size": 0.3,
    "random_state": 42,
    "hyperparameters": {"n_neighbors": 7}
  }'
```

约束：

- 超参数字段名为 `hyperparameters`（不是 `params`），未知字段会被忽略
- `test_size` 仅允许 `0.2` / `0.3` / `0.4`，默认 `0.3`
- 数据集与模型的任务类型必须匹配，否则实验以 `failed` 结束并在 `error` 中给出原因
- `random_state` 由实验顶层统一控制，不在参数表单中重复出现

## 数据集与模型

内置数据集：

| 键 | 任务类型 | 样本数 | 特征数 |
| --- | --- | --- | --- |
| `iris` | classification | 150 | 4 |
| `wine` | classification | 178 | 13 |
| `breast_cancer` | classification | 569 | 30 |
| `digits` | classification | 1797 | 64 |
| `diabetes` | regression | 442 | 10 |

内置模型：

| 键 | 任务类型 |
| --- | --- |
| `knn_classifier` | classification |
| `gaussian_nb` | classification |
| `logistic_regression` | classification |
| `linear_svm` | classification |
| `mlp_classifier` | classification |
| `linear_regression` | regression |
| `ridge_regression` | regression |
| `decision_tree_regressor` | regression |
| `random_forest_regressor` | regression |
| `gbdt_regressor` | regression |

## 结果结构

实验状态为 `completed` 时，响应中的 `evaluation` 字段才有值，且按任务类型填充：

- **分类**：`metrics`（`accuracy`、`precision_weighted`、`recall_weighted`、`f1_weighted`）+ `confusion_matrix` + `class_labels`；`y_true` / `y_pred` 为 `null`
- **回归**：`metrics`（`mse`、`rmse`、`mae`、`r2`）+ `y_true` / `y_pred`（等长数组）；`confusion_matrix` / `class_labels` 为 `null`

`training_info` 为可选训练元数据（`loss_history`、`n_iter`、`stop_reason`、`coefficients`、`intercept`），仅迭代类模型提供，前端按存在字段渲染。

## 配置

配置来自环境变量或 `.env`，前缀统一为 `MLLAB_`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MLLAB_APP_NAME` | `MLLab` | 应用名称 |
| `MLLAB_ENVIRONMENT` | `development` | 运行环境 |
| `MLLAB_DEBUG` | `true` | 调试模式 |
| `MLLAB_API_PREFIX` | `/api/v1` | 接口前缀 |
| `MLLAB_DATABASE_URL` | `sqlite:///<项目根>/mllab.db` | 数据库地址，默认落在项目根目录 |
| `MLLAB_EXPERIMENTS_DIR` | `<项目根>/experiments` | 实验产物目录 |
| `MLLAB_CORS_ORIGINS` | `["http://localhost:5173","http://127.0.0.1:5173"]` | 允许的前端来源 |

数据库与产物目录默认基于项目根解析为绝对路径，与启动时的工作目录无关。

## 测试与代码规范

```bash
pytest                 # 全量测试
ruff check .           # 静态检查
cd frontend && npm run lint && npm run build
```

## 设计说明

- **分层解耦**：`backend` 通过 Protocol 调用 `ml_engine`，算法层可独立测试与替换
- **数据泄漏防护**：先切分训练/测试集，标准化器仅在训练集上 `fit`
- **参数元信息驱动**：超参数表单由 `/api/v1/models` 的 `parameters` 字段动态生成，新增模型无需改前端
- **产物隔离**：每次实验写入独立的 `experiments/<实验 id>/` 目录，删除实验时一并清理

## 当前范围

- 仅支持上述 5 个内置数据集，暂不提供 CSV 上传
- 任务类型限定为分类与回归，聚类不在本阶段范围
- 实验通过 FastAPI `BackgroundTasks` 在同进程执行，适合单机演示；生产环境应替换为独立任务队列
