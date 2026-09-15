# MLLab

MLLab 是一个面向课程项目的可扩展机器学习实验平台。项目将系统层与算法层分离，后续可以在不修改核心 API 的情况下接入新的数据集、模型、评价指标与可视化结果。

## 当前状态

项目初始化已经完成：

- FastAPI 应用入口与版本化路由
- 环境变量配置
- `backend` / `ml_engine` 分层目录
- 健康检查接口
- 基础自动化测试
- 实验输出目录占位

模型注册、数据集注册、实验数据库及执行器将在后续任务中实现。

## 环境要求

- Python 3.10+
- 推荐使用 Python 3.11

## 本地启动

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[ml,dev]"
cp .env.example .env
uvicorn backend.main:app --reload
```

打开：

- 服务首页：<http://127.0.0.1:8000/>
- 健康检查：<http://127.0.0.1:8000/api/v1/health>
- API 文档：<http://127.0.0.1:8000/docs>

## 运行测试

```bash
pytest
```

## 项目结构

```text
MLLab/
├── backend/              # API、配置、数据库与业务编排
│   ├── api/              # HTTP 路由
│   ├── core/             # 全局配置与基础设施
│   ├── database/         # 数据库模型与会话（待实现）
│   └── services/         # 实验执行服务（待实现）
├── ml_engine/            # 与 Web 框架解耦的机器学习能力
│   ├── datasets/         # 数据集接口与注册器（待实现）
│   ├── evaluation/       # 统一评价接口（待实现）
│   ├── models/           # 模型基类与注册器（待实现）
│   └── visualization/    # 可视化结果协议（待实现）
├── experiments/          # 本地实验产物，不提交 Git
├── frontend/             # 前端项目预留目录
├── tests/                # 自动化测试
├── .env.example          # 配置模板
└── pyproject.toml        # 依赖和工具配置
```

## 分支建议

```text
feature/d-backend -> develop -> main
```

首个提交建议：

```bash
git add .
git commit -m "feat: initialize project architecture"
```

