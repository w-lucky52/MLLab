<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import {
  ArrowLeft,
  DataAnalysis,
  Download,
  RefreshRight,
  TrendCharts,
} from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

const experiment = ref(null)
const metricChartRef = ref(null)
const matrixChartRef = ref(null)

let metricChart = null
let matrixChart = null
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

const resultOverviewTitle = computed(() => {
  if (!hasResult.value) return '实验运行中'

  if (isRegression.value) {
    const r2 = metrics.value[3]?.value || 0
    return r2 >= 0.7 ? '模型拟合表现良好' : '模型实验已完成'
  }

  const accuracy = metrics.value[0]?.value || 0
  return accuracy >= 90 ? '模型整体表现良好' : '模型实验已完成'
})

const resultOverviewText = computed(() => {
  if (!hasResult.value) {
    return '正在等待后端返回实验结果。'
  }

  if (isRegression.value) {
    return `本次真实实验中，模型的 MAE 为 ${metrics.value[0]?.value}，RMSE 为 ${metrics.value[2]?.value}，R² 为 ${metrics.value[3]?.value}。可继续调整参数或更换算法进行对比实验。`
  }

  return `本次真实实验中，模型准确率为 ${metrics.value[0]?.value}%，加权精确率为 ${metrics.value[1]?.value}%，加权召回率为 ${metrics.value[2]?.value}%，加权 F1 为 ${metrics.value[3]?.value}%。可继续调整参数或更换算法进行对比实验。`
})
const initMetricChart = () => {
  if (!metricChartRef.value || !hasResult.value) return

  metricChart?.dispose()
  metricChart = echarts.init(metricChartRef.value)

  metricChart.setOption({
    animationDuration: 900,
    grid: {
      top: 25,
      right: 18,
      bottom: 35,
      left: 58,
    },
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const item = Array.isArray(params) ? params[0] : params
        const unit = isRegression.value ? '' : '%'
        return `${item.name}<br/>${item.seriesName}：${toNumber(item.value).toFixed(2)}${unit}`
      },
    },
    xAxis: {
      type: 'category',
      data: metrics.value.map((item) => item.name),
      axisTick: {
        show: false,
      },
      axisLine: {
        lineStyle: {
          color: '#dbe3ef',
        },
      },
      axisLabel: {
        color: '#64748b',
      },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: isRegression.value ? undefined : 100,
      axisLabel: {
        color: '#94a3b8',
        formatter: (value) => {
          return isRegression.value ? value : `${value}%`
        },
      },
      splitLine: {
        lineStyle: {
          color: '#edf2f7',
        },
      },
    },
    series: [
      {
        name: '评估指标',
        type: 'bar',
        barWidth: 38,
        data: metrics.value.map((item) => item.value),
        itemStyle: {
          borderRadius: [9, 9, 2, 2],
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            {
              offset: 0,
              color: '#6366f1',
            },
            {
              offset: 1,
              color: '#3b82f6',
            },
          ]),
        },
      },
    ],
  })
}

const initClassificationChart = () => {
  const matrix = evaluation.value?.confusion_matrix || []
  const labels = evaluation.value?.class_labels || []

  const matrixData = matrix.flatMap((row, rowIndex) =>
    row.map((value, columnIndex) => [columnIndex, rowIndex, value]),
  )

  const displayLabels = labels.length
    ? labels.map((label) => String(label))
    : matrix.map((_, index) => `类别 ${index + 1}`)

  const maximum = Math.max(1, ...matrixData.map((item) => toNumber(item[2])))

  matrixChart.setOption({
    animationDuration: 900,
    tooltip: {
      position: 'top',
      formatter: (params) => {
        const predicted = displayLabels[params.value[0]]
        const actual = displayLabels[params.value[1]]

        return `真实类别：${actual}<br/>预测类别：${predicted}<br/>样本数：${params.value[2]}`
      },
    },
    grid: {
      top: 30,
      right: 30,
      bottom: 65,
      left: 70,
    },
    xAxis: {
      type: 'category',
      data: displayLabels,
      name: '预测类别',
      nameLocation: 'middle',
      nameGap: 38,
      splitArea: {
        show: true,
      },
      axisLabel: {
        color: '#64748b',
      },
    },
    yAxis: {
      type: 'category',
      data: displayLabels,
      name: '真实类别',
      nameLocation: 'middle',
      nameGap: 50,
      splitArea: {
        show: true,
      },
      axisLabel: {
        color: '#64748b',
      },
    },
    visualMap: {
      min: 0,
      max: maximum,
      show: false,
      inRange: {
        color: ['#eff6ff', '#93c5fd', '#2563eb'],
      },
    },
    series: [
      {
        name: '混淆矩阵',
        type: 'heatmap',
        data: matrixData,
        label: {
          show: true,
          color: '#0f172a',
          fontWeight: 700,
        },
      },
    ],
  })
}

const initRegressionChart = () => {
  const yTrue = evaluation.value?.y_true || []
  const yPred = evaluation.value?.y_pred || []

  const scatterData = yTrue.map((value, index) => [toNumber(value), toNumber(yPred[index])])

  const allValues = scatterData.flat()
  const minimum = Math.min(...allValues)
  const maximum = Math.max(...allValues)

  matrixChart.setOption({
    animationDuration: 900,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        return `真实值：${params.value[0].toFixed(2)}<br/>预测值：${params.value[1].toFixed(2)}`
      },
    },
    grid: {
      top: 30,
      right: 30,
      bottom: 60,
      left: 70,
    },
    xAxis: {
      type: 'value',
      name: '真实值',
      nameLocation: 'middle',
      nameGap: 38,
      splitLine: {
        lineStyle: {
          color: '#edf2f7',
        },
      },
    },
    yAxis: {
      type: 'value',
      name: '预测值',
      nameLocation: 'middle',
      nameGap: 50,
      splitLine: {
        lineStyle: {
          color: '#edf2f7',
        },
      },
    },
    series: [
      {
        name: '预测样本',
        type: 'scatter',
        symbolSize: 10,
        data: scatterData,
        itemStyle: {
          color: '#3b82f6',
          opacity: 0.75,
        },
      },
      {
        name: '理想预测线',
        type: 'line',
        data: [
          [minimum, minimum],
          [maximum, maximum],
        ],
        symbol: 'none',
        lineStyle: {
          color: '#ef4444',
          type: 'dashed',
          width: 2,
        },
      },
    ],
  })
}

const initResultChart = () => {
  if (!matrixChartRef.value || !hasResult.value) return

  matrixChart?.dispose()
  matrixChart = echarts.init(matrixChartRef.value)

  if (isRegression.value) {
    initRegressionChart()
  } else {
    initClassificationChart()
  }
}

const initCharts = async () => {
  if (!hasResult.value) return

  await nextTick()
  initMetricChart()
  initResultChart()
}

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
      await initCharts()
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

const resizeCharts = () => {
  metricChart?.resize()
  matrixChart?.resize()
}

const exportResult = () => {
  ElMessage.info('结果导出功能将在后续步骤接入')
}

onMounted(() => {
  fetchExperiment()
  window.addEventListener('resize', resizeCharts)
})

watch(
  () => route.params.id,
  () => {
    experiment.value = null
    metricChart?.dispose()
    matrixChart?.dispose()
    fetchExperiment()
  },
)

onBeforeUnmount(() => {
  clearTimeout(pollTimer)
  window.removeEventListener('resize', resizeCharts)
  metricChart?.dispose()
  matrixChart?.dispose()
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
            <small>相比基准模型</small>
          </div>
        </el-card>
      </section>

      <section class="chart-grid">
        <el-card class="chart-card metric-chart-card" shadow="never">
          <template #header>
            <div class="card-header">
              <div>
                <h3>模型评估指标</h3>
                <p>
                  {{ isRegression ? '各项回归指标的综合表现' : '各项分类指标的综合表现' }}
                </p>
              </div>

              <el-tag effect="plain" round>
                {{ isRegression ? '回归结果' : '百分比' }}
              </el-tag>
            </div>
          </template>

          <div ref="metricChartRef" class="chart"></div>
        </el-card>

        <el-card class="chart-card" shadow="never">
          <template #header>
            <div class="card-header">
              <div>
                <h3>{{ isRegression ? '真实值与预测值' : '混淆矩阵' }}</h3>
                <p>
                  {{
                    isRegression ? '预测结果与真实目标值的对应关系' : '真实类别与预测类别的对应关系'
                  }}
                </p>
              </div>

              <el-tag effect="plain" round>
                {{ isRegression ? '回归结果' : '分类结果' }}
              </el-tag>
            </div>
          </template>

          <div ref="matrixChartRef" class="chart"></div>
        </el-card>
      </section>

      <section class="analysis-card">
        <div class="analysis-icon">
          <el-icon><TrendCharts /></el-icon>
        </div>

        <div>
          <span>结果概览</span>
          <h3>{{ resultOverviewTitle }}</h3>
          <p>{{ resultOverviewText }}</p>
        </div>
      </section>

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

.result-alert {
  margin-top: 22px;
  border-radius: 13px;
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

.summary-icon,
.analysis-icon {
  display: grid;
  flex: none;
  place-items: center;
}

.summary-icon {
  width: 48px;
  height: 48px;
  color: #059669;
  font-size: 22px;
  border-radius: 14px;
  background: #ecfdf5;
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

.metric-footer small {
  color: #94a3b8;
}

.chart-grid {
  display: grid;
  grid-template-columns: 1.25fr 0.75fr;
  gap: 18px;
  margin-top: 20px;
}

.chart-grid.single-chart {
  grid-template-columns: 1fr;
}
.chart-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.chart-card :deep(.el-card__body) {
  padding: 5px 16px 15px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.card-header h3 {
  margin: 0 0 6px;
  color: #1e293b;
  font-size: 15px;
}

.card-header p {
  margin: 0;
  color: #94a3b8;
  font-size: 11px;
}

.chart {
  width: 100%;
  height: 330px;
}

.analysis-card {
  display: flex;
  align-items: flex-start;
  gap: 17px;
  padding: 22px 24px;
  margin-top: 20px;
  border: 1px solid #dbeafe;
  border-radius: 17px;
  background: linear-gradient(135deg, #eff6ff, #f8faff);
}

.analysis-icon {
  width: 46px;
  height: 46px;
  color: #2563eb;
  font-size: 21px;
  border-radius: 14px;
  background: white;
  box-shadow: 0 8px 20px rgb(37 99 235 / 10%);
}

.analysis-card span {
  color: #3b82f6;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.2px;
}

.analysis-card h3 {
  margin: 5px 0 7px;
  color: #1e3a8a;
  font-size: 16px;
}

.analysis-card p {
  margin: 0;
  color: #64748b;
  font-size: 12px;
  line-height: 1.7;
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
  .metrics-grid,
  .chart-grid {
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
