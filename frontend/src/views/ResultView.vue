<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft, DataAnalysis, Download, RefreshRight, TrendCharts } from '@element-plus/icons-vue'

import ClassificationPerformance from '@/components/result/ClassificationPerformance.vue'
import ModelExplanation from '@/components/result/ModelExplanation.vue'
import RegressionPerformance from '@/components/result/RegressionPerformance.vue'
import TrainingProcess from '@/components/result/TrainingProcess.vue'

const route = useRoute()
const router = useRouter()

const experiment = ref(null)
const activeTab = ref('performance')
let pollTimer = null

const datasetNameMap = {
  iris: '鸢尾花数据集',
  wine: '葡萄酒数据集',
  breast_cancer: '乳腺癌数据集',
  digits: '手写数字数据集',
  diabetes: '糖尿病数据集',
}

const algorithmNameMap = {
  knn_classifier: 'K 近邻（KNN）',
  gaussian_nb: '高斯朴素贝叶斯',
  logistic_regression: '逻辑回归',
  linear_svm: '线性 SVM',
  mlp_classifier: '多层感知机',
  linear_regression: '线性回归',
  ridge_regression: '岭回归',
  decision_tree_regressor: '决策树回归（CART）',
  random_forest_regressor: '随机森林回归',
  gbdt_regressor: '梯度提升树回归（GBDT）',
}

const datasetName = computed(() => {
  const name = experiment.value?.dataset_name
  return datasetNameMap[name] || name || '加载中'
})

const algorithmName = computed(() => {
  const name = experiment.value?.model_name
  return algorithmNameMap[name] || name || '加载中'
})

const evaluation = computed(() => {
  return experiment.value?.evaluation || null
})

const hasResult = computed(() => {
  return experiment.value?.status === 'completed' && Boolean(evaluation.value)
})

const isRegression = computed(() => {
  return (
    experiment.value?.task_type === 'regression' || evaluation.value?.task_type === 'regression'
  )
})

const toNumber = (value) => {
  const result = Number(value)
  return Number.isFinite(result) ? result : 0
}

const toPercentage = (value) => {
  const result = toNumber(value)
  return result <= 1 ? result * 100 : result
}

const roundValue = (value, digits = 2) => {
  return Number(toNumber(value).toFixed(digits))
}

const metrics = computed(() => {
  const values = evaluation.value?.metrics || {}

  if (isRegression.value) {
    return [
      {
        name: 'MAE',
        value: roundValue(values.mae),
        change: '越低越好',
        color: 'blue',
        unit: '',
      },
      {
        name: 'MSE',
        value: roundValue(values.mse),
        change: '越低越好',
        color: 'purple',
        unit: '',
      },
      {
        name: 'RMSE',
        value: roundValue(values.rmse),
        change: '越低越好',
        color: 'green',
        unit: '',
      },
      {
        name: 'R²',
        value: roundValue(values.r2, 4),
        change: '越高越好',
        color: 'orange',
        unit: '',
      },
    ]
  }
  return [
    {
      name: '准确率',
      value: roundValue(toPercentage(values.accuracy)),
      change: '越高越好',
      color: 'blue',
      unit: '%',
    },
    {
      name: '加权精确率',
      value: roundValue(toPercentage(values.precision_weighted)),
      change: '越高越好',
      color: 'purple',
      unit: '%',
    },
    {
      name: '加权召回率',
      value: roundValue(toPercentage(values.recall_weighted)),
      change: '越高越好',
      color: 'green',
      unit: '%',
    },
    {
      name: '加权 F1',
      value: roundValue(toPercentage(values.f1_weighted)),
      change: '越高越好',
      color: 'orange',
      unit: '%',
    },
  ]
})

const fetchExperiment = async () => {
  clearTimeout(pollTimer)

  try {
    const response = await fetch(`/api/v1/experiments/${route.params.id}`)
    const data = await response.json()

    if (!response.ok) {
      const message = typeof data.detail === 'string' ? data.detail : '获取实验结果失败'

      throw new Error(message)
    }

    experiment.value = data

    if (data.status === 'completed') {
      return
    }

    if (data.status === 'failed') {
      ElMessage.error(data.error || '实验运行失败')
      return
    }

    pollTimer = setTimeout(fetchExperiment, 1500)
  } catch (error) {
    ElMessage.error(error.message || '无法获取实验结果')
  }
}

const exportResult = () => {
  ElMessage.info('结果导出功能将在后续步骤接入')
}

onMounted(() => {
  fetchExperiment()
})

watch(
  () => route.params.id,
  () => {
    experiment.value = null
    activeTab.value = 'performance'
    fetchExperiment()
  },
)

onBeforeUnmount(() => {
  clearTimeout(pollTimer)
})
</script>

<template>
  <div class="result-page">
    <section class="page-heading">
      <div>
        <span class="eyebrow">MODEL INSIGHTS</span>
        <h1>实验结果分析</h1>
        <p>通过核心指标和可视化图表分析模型表现。</p>
      </div>

      <div class="heading-icon">
        <el-icon><TrendCharts /></el-icon>
      </div>
    </section>

    <el-empty v-if="!hasResult" class="empty-result" description="暂无可展示的实验结果">
      <el-button type="primary" @click="router.push('/data')"> 开始新实验 </el-button>
    </el-empty>

    <template v-else>
      <section class="experiment-summary">
        <div class="summary-main">
          <div class="summary-icon">
            <el-icon><DataAnalysis /></el-icon>
          </div>

          <div>
            <span>本次实验</span>
            <h3>{{ algorithmName }}</h3>
            <p>{{ datasetName }}</p>
          </div>
        </div>

        <div class="summary-item">
          <span>测试集比例</span>
          <strong>{{ Math.round(Number(experiment?.test_size ?? 0.3) * 100) }}% </strong>
        </div>

        <div class="summary-item">
          <span>随机种子</span>
          <strong>{{ experiment?.random_state ?? 42 }}</strong>
        </div>

        <div class="summary-item">
          <span>运行状态</span>
          <el-tag type="success" effect="light" round> 已完成 </el-tag>
        </div>
      </section>

      <section class="metrics-grid">
        <el-card
          v-for="metric in metrics"
          :key="metric.name"
          :class="['metric-card', metric.color]"
          shadow="never"
        >
          <div class="metric-header">
            <span>{{ metric.name }}</span>
            <div class="metric-dot"></div>
          </div>

          <div class="metric-value">
            {{ metric.value }}
            <small>{{ metric.unit }}</small>
          </div>

          <div class="metric-footer">
            <span>{{ metric.change }}</span>
          </div>
        </el-card>
      </section>

      <el-tabs v-model="activeTab" class="result-tabs">
        <el-tab-pane :label="isRegression ? '回归表现' : '分类表现'" name="performance">
          <RegressionPerformance
            v-if="isRegression"
            :y-true="evaluation?.y_true"
            :y-pred="evaluation?.y_pred"
            :active="activeTab === 'performance'"
          />

          <ClassificationPerformance
            v-else
            :evaluation="evaluation"
            :active="activeTab === 'performance'"
          />
        </el-tab-pane>

        <el-tab-pane label="训练过程" name="training">
          <TrainingProcess
            :model-name="experiment?.model_name || ''"
            :hyperparameters="experiment?.hyperparameters"
            :training-info="experiment?.training_info"
            :active="activeTab === 'training'"
          />
        </el-tab-pane>

        <el-tab-pane label="模型解释" name="interpretation">
          <ModelExplanation
            :model-name="experiment?.model_name || ''"
            :task-type="isRegression ? 'regression' : 'classification'"
            :training-info="experiment?.training_info"
            :score-classes="evaluation?.score_classes"
            :score-class-names="evaluation?.score_class_names"
            :active="activeTab === 'interpretation'"
          />
        </el-tab-pane>
      </el-tabs>

      <footer class="page-actions">
        <el-button
          size="large"
          @click="
            router.push({
              path: '/experiment',
              query: { dataset: experiment?.dataset_name },
            })
          "
        >
          <el-icon><ArrowLeft /></el-icon>
          修改实验参数
        </el-button>

        <div>
          <el-button size="large" @click="exportResult">
            <el-icon><Download /></el-icon>
            导出结果
          </el-button>

          <el-button type="primary" size="large" @click="router.push('/data')">
            <el-icon><RefreshRight /></el-icon>
            开始新实验
          </el-button>
        </div>
      </footer>
    </template>
  </div>
</template>

<style scoped>
.result-page {
  padding-bottom: 20px;
}

.page-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 30px 34px;
  color: white;
  border-radius: 20px;
  background:
    radial-gradient(circle at 82% 10%, rgb(52 211 153 / 25%), transparent 30%),
    linear-gradient(120deg, #064e3b, #047857);
  box-shadow: 0 20px 45px rgb(5 150 105 / 14%);
}

.eyebrow {
  color: #6ee7b7;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 1.8px;
}

.page-heading h1 {
  margin: 8px 0;
  font-size: 30px;
}

.page-heading p {
  margin: 0;
  color: #d1fae5;
}

.heading-icon {
  display: grid;
  width: 72px;
  height: 72px;
  font-size: 32px;
  border: 1px solid rgb(255 255 255 / 13%);
  border-radius: 22px;
  background: rgb(255 255 255 / 9%);
  place-items: center;
}

.empty-result {
  min-height: 430px;
}

.experiment-summary {
  display: grid;
  grid-template-columns: 1.8fr repeat(3, 1fr);
  align-items: center;
  margin-top: 20px;
  overflow: hidden;
  border: 1px solid #e5eaf2;
  border-radius: 17px;
  background: white;
}

.summary-main,
.summary-item {
  padding: 20px 22px;
}

.summary-main {
  display: flex;
  align-items: center;
  gap: 14px;
}

.summary-icon {
  display: grid;
  width: 48px;
  height: 48px;
  flex: none;
  color: #059669;
  font-size: 22px;
  border-radius: 14px;
  background: #ecfdf5;
  place-items: center;
}

.summary-main span,
.summary-item span {
  color: #94a3b8;
  font-size: 11px;
}

.summary-main h3 {
  margin: 4px 0;
  color: #0f172a;
  font-size: 15px;
}

.summary-main p {
  margin: 0;
  color: #64748b;
  font-size: 12px;
}

.summary-item {
  display: flex;
  min-height: 68px;
  flex-direction: column;
  justify-content: center;
  gap: 8px;
  border-left: 1px solid #edf1f6;
}

.summary-item strong {
  color: #334155;
  font-size: 15px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 17px;
  margin-top: 20px;
}

.metric-card {
  position: relative;
  overflow: hidden;
}

.metric-card :deep(.el-card__body) {
  padding: 21px;
}

.metric-card::after {
  position: absolute;
  top: -30px;
  right: -30px;
  width: 90px;
  height: 90px;
  content: '';
  border-radius: 50%;
  opacity: 0.1;
}

.metric-card.blue::after {
  background: #2563eb;
}

.metric-card.purple::after {
  background: #7c3aed;
}

.metric-card.green::after {
  background: #059669;
}

.metric-card.orange::after {
  background: #ea580c;
}

.metric-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #64748b;
  font-size: 12px;
}

.metric-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.blue .metric-dot {
  background: #2563eb;
}

.purple .metric-dot {
  background: #7c3aed;
}

.green .metric-dot {
  background: #059669;
}

.orange .metric-dot {
  background: #ea580c;
}

.metric-value {
  margin: 13px 0 10px;
  color: #0f172a;
  font-size: 31px;
  font-weight: 750;
}

.metric-value small {
  color: #64748b;
  font-size: 14px;
}

.metric-footer {
  display: flex;
  gap: 7px;
  font-size: 11px;
}

.metric-footer span {
  color: #059669;
  font-weight: 700;
}

.result-tabs {
  margin-top: 22px;
}

.result-tabs :deep(.el-tabs__header) {
  margin-bottom: 0;
}

.result-tabs :deep(.el-tabs__item) {
  font-size: 14px;
  font-weight: 600;
}

.result-tabs :deep(.el-tabs__nav-wrap::after) {
  height: 1px;
  background-color: #e8edf5;
}

.page-actions {
  display: flex;
  justify-content: space-between;
  margin-top: 22px;
}

.page-actions > div {
  display: flex;
  gap: 10px;
}

@media (max-width: 1000px) {
  .experiment-summary,
  .metrics-grid {
    grid-template-columns: 1fr;
  }

  .summary-item {
    border-top: 1px solid #edf1f6;
    border-left: 0;
  }
}

@media (max-width: 700px) {
  .heading-icon {
    display: none;
  }

  .page-actions,
  .page-actions > div {
    align-items: stretch;
    flex-direction: column;
    gap: 10px;
  }
}
</style>
