<script setup>
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from 'vue'
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

const metricChartRef = ref(null)
const matrixChartRef = ref(null)

let metricChart = null
let matrixChart = null

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
  linear_regression: '线性回归',
  ridge_regression: '岭回归',
  decision_tree_regressor: '决策树回归（CART）',
  random_forest_regressor: '随机森林回归',
  gbdt_regressor: '梯度提升树回归（GBDT）',
}

const datasetName = computed(() => {
  return datasetNameMap[route.query.dataset]
    || route.query.dataset
    || '未选择'
})

const algorithmName = computed(() => {
  return algorithmNameMap[route.query.algorithm]
    || route.query.algorithm
    || '未选择'
})

const hasResult = computed(() => {
  return Boolean(route.query.dataset && route.query.algorithm)
})

const regressionAlgorithms = [
  'linear_regression',
  'ridge_regression',
  'decision_tree_regressor',
  'random_forest_regressor',
  'gbdt_regressor',
]

const isRegression = computed(() => {
  return (
    route.query.dataset === 'diabetes' ||
    regressionAlgorithms.includes(String(route.query.algorithm || ''))
  )
})

const metrics = computed(() => {
  if (isRegression.value) {
    return [
      {
        name: 'MAE',
        value: 42.36,
        change: '越低越好',
        color: 'blue',
        unit: '',
      },
      {
        name: 'MSE',
        value: 2856.42,
        change: '越低越好',
        color: 'purple',
        unit: '',
      },
      {
        name: 'RMSE',
        value: 53.45,
        change: '越低越好',
        color: 'green',
        unit: '',
      },
      {
        name: 'R²',
        value: 0.48,
        change: '越高越好',
        color: 'orange',
        unit: '',
      },
    ]
  }

  return [
    {
      name: '准确率',
      value: 94.67,
      change: '+2.31%',
      color: 'blue',
      unit: '%',
    },
    {
      name: '加权精确率',
      value: 94.31,
      change: '+1.86%',
      color: 'purple',
      unit: '%',
    },
    {
      name: '加权召回率',
      value: 93.85,
      change: '+1.52%',
      color: 'green',
      unit: '%',
    },
    {
      name: '加权 F1',
      value: 94.08,
      change: '+1.73%',
      color: 'orange',
      unit: '%',
    },
  ]
})

const initMetricChart = () => {
  if (!metricChartRef.value) return

  metricChart?.dispose()
  metricChart = echarts.init(metricChartRef.value)

  metricChart.setOption({
    animationDuration: 900,
    grid: {
      top: 25,
      right: 18,
      bottom: 35,
      left: 48,
    },
    tooltip: {
      trigger: 'axis',
     formatter: (params) => {
  const item = Array.isArray(params) ? params[0] : params
  const unit = isRegression.value ? '' : '%'
  return `${item.name}<br/>${item.seriesName}：${item.value}${unit}`
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
  min: isRegression.value ? 0 : 80,
  max: isRegression.value ? null : 100,
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

const initMatrixChart = () => {
  if (!matrixChartRef.value) return

  matrixChart?.dispose()
  matrixChart = echarts.init(matrixChartRef.value)

  const matrixData = [
    [0, 0, 10],
    [1, 0, 0],
    [2, 0, 0],
    [0, 1, 0],
    [1, 1, 9],
    [2, 1, 1],
    [0, 2, 0],
    [1, 2, 1],
    [2, 2, 9],
  ]

  matrixChart.setOption({
    animationDuration: 900,
    tooltip: {
      position: 'top',
      formatter: (params) => {
        return `真实类别 ${params.value[1] + 1}<br/>
                预测类别 ${params.value[0] + 1}：${params.value[2]}`
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
      data: ['类别 1', '类别 2', '类别 3'],
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
      data: ['类别 1', '类别 2', '类别 3'],
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
      max: 10,
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
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgb(37 99 235 / 30%)',
          },
        },
      },
    ],
  })
}

const initCharts = async () => {
  if (!hasResult.value) return

  await nextTick()
  initMetricChart()
  initMatrixChart()
}

const resizeCharts = () => {
  metricChart?.resize()
  matrixChart?.resize()
}

const exportResult = () => {
  ElMessage.success('实验结果导出功能将在连接后端后启用')
}

onMounted(() => {
  initCharts()
  window.addEventListener('resize', resizeCharts)
})

watch(
  () => route.fullPath,
  () => {
    initCharts()
  },
)

onBeforeUnmount(() => {
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

    <el-empty
      v-if="!hasResult"
      class="empty-result"
      description="暂无可展示的实验结果"
    >
      <el-button type="primary" @click="router.push('/data')">
        开始新实验
      </el-button>
    </el-empty>

    <template v-else>
      <el-alert
        title="演示数据说明"
        description="当前评估指标与混淆矩阵为前端模拟数据，连接后端接口后将展示真实模型结果。"
        type="warning"
        :closable="false"
        show-icon
        class="result-alert"
      />

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
          <strong>
            {{ Math.round(Number(route.query.testSize || 0.2) * 100) }}%
          </strong>
        </div>

        <div class="summary-item">
          <span>随机种子</span>
          <strong>{{ route.query.randomState || 42 }}</strong>
        </div>

        <div class="summary-item">
          <span>运行状态</span>
          <el-tag type="success" effect="light" round>
            已完成
          </el-tag>
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

     <section
  class="chart-grid"
  :class="{ 'single-chart': isRegression }"
>
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

  <el-card
    v-if="!isRegression"
    class="chart-card"
    shadow="never"
  >
    <template #header>
      <div class="card-header">
        <div>
          <h3>混淆矩阵</h3>
          <p>真实类别与预测类别的对应关系</p>
        </div>

        <el-tag effect="plain" round>分类结果</el-tag>
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
          <h3>模型整体表现良好</h3>
          <p>
            当前模拟结果中，模型准确率达到 94.67%，各项指标较为均衡。
            后续可通过调整参数或更换算法进行对比实验。
          </p>
        </div>
      </section>

      <footer class="page-actions">
        <el-button
          size="large"
          @click="router.push({
            path: '/experiment',
            query: { dataset: route.query.dataset },
          })"
        >
          <el-icon><ArrowLeft /></el-icon>
          修改实验参数
        </el-button>

        <div>
          <el-button size="large" @click="exportResult">
            <el-icon><Download /></el-icon>
            导出结果
          </el-button>

          <el-button
            type="primary"
            size="large"
            @click="router.push('/data')"
          >
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
  content: "";
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
