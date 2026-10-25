<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { formatNumber } from '@/utils/training'
import {
  classDisplayLookup,
  classDisplayName,
} from '@/utils/classDisplay'
import {
  classNameNotice,
  featureNameNotice,
  heatmapCells,
  summarizeMulticlassCoefficients,
  symmetricRange,
} from '@/utils/modelExplanation'

const INTERCEPT_LABEL = '截距'

const props = defineProps({
  coefficients: { type: [Array, Object], default: null },
  featureNames: { type: Array, default: null },
  classes: { type: Array, default: null },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
  intercept: { type: [Array, Number], default: null },
  modelName: { type: String, default: 'logistic_regression' },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const summary = computed(() =>
  summarizeMulticlassCoefficients({
    coefficients: props.coefficients,
    featureNames: props.featureNames,
    classes: props.classes,
    scoreClasses: props.scoreClasses,
    intercept: props.intercept,
  }),
)

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const hasData = computed(() => summary.value.available)

const classNotice = computed(() => classNameNotice(summary.value.classNames))
const featureNotice = computed(() => featureNameNotice(summary.value.featureNames))

const isLogistic = computed(() => props.modelName === 'logistic_regression')

const cardTitle = computed(() => (isLogistic.value ? '类别 × 特征系数' : '类别 × 特征决策分数系数'))

const cardSubtitle = computed(() =>
  isLogistic.value
    ? '颜色以 0 为中心，展示各特征对对应类别得分（logit）的影响方向'
    : '颜色以 0 为中心，展示各特征对对应类别决策分数的影响方向',
)

// Row order stays untouched; only the class labels are swapped for display names.
const displayClassNames = computed(() =>
  summary.value.classNames.names.map((name) => classDisplayName(name, displayLookup.value)),
)

const columnLabels = computed(() => {
  const names = summary.value.featureNames.names
  return summary.value.interceptAppended ? [...names, INTERCEPT_LABEL] : names
})

const cells = computed(() => heatmapCells(summary.value.heatmapMatrix))
const range = computed(() => symmetricRange(summary.value.heatmapMatrix))

const showZoom = computed(() => columnLabels.value.length > 10)

const buildOption = () => {
  const data = cells.value
  const scale = range.value
  const rows = data?.cells || []
  const classLabels = displayClassNames.value
  const featureLabels = columnLabels.value
  const cellCount = (summary.value.classCount ?? 0) * featureLabels.length
  const showLabels = cellCount <= 24

  return {
    animationDuration: 500,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const [columnIndex, rowIndex, value] = params.value
        return `类别：${classLabels[rowIndex] ?? ''}<br/>特征：${featureLabels[columnIndex] ?? ''}<br/>系数：${formatNumber(value)}`
      },
    },
    grid: {
      top: 16,
      right: 24,
      bottom: showZoom.value ? 92 : 64,
      left: 12,
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: featureLabels,
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
      axisLabel: {
        color: '#64748b',
        fontSize: 11,
        interval: 0,
        rotate: featureLabels.length > 8 ? 35 : 0,
        width: 96,
        overflow: 'truncate',
      },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: classLabels,
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
      axisLabel: { color: '#475569', fontSize: 11, width: 92, overflow: 'truncate' },
    },
    visualMap: {
      min: scale?.min ?? -1,
      max: scale?.max ?? 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 4,
      itemWidth: 12,
      itemHeight: 110,
      text: ['正系数', '负系数'],
      textStyle: { color: '#94a3b8', fontSize: 10 },
      inRange: { color: ['#dc2626', '#f8fafc', '#059669'] },
    },
    dataZoom: showZoom.value
      ? [
          { type: 'inside', xAxisIndex: 0 },
          { type: 'slider', xAxisIndex: 0, height: 14, bottom: 34 },
        ]
      : [],
    series: [
      {
        type: 'heatmap',
        data: rows,
        label: {
          show: showLabels,
          fontSize: 10,
          color: '#334155',
          formatter: (params) => formatNumber(params.value[2]),
        },
        itemStyle: { borderColor: '#ffffff', borderWidth: 1 },
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(15, 23, 42, 0.18)' } },
      },
    ],
  }
}

const { mount } = useChart(
  chartEl,
  buildOption,
  () => props.active && hasData.value,
)

useExportableChart({
  id: 'multiclass-coefficient-heatmap',
  section: '模型洞察',
  title: '类别系数热力图',
  order: 2,
  available: () => hasData.value,
  buildOption,
})

watch(
  () => [
    props.coefficients,
    props.featureNames,
    props.classes,
    props.scoreClasses,
    props.scoreClassNames,
    props.intercept,
  ],
  () => {
    mount()
  },
  { deep: true },
)
</script>

<template>
  <el-card class="chart-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>{{ cardTitle }}</h3>
          <p>{{ cardSubtitle }}</p>
        </div>
        <el-tag effect="plain" round>{{ isLogistic ? '逻辑回归' : '线性 SVM' }}</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty compact">
      <el-empty
        :image-size="56"
        :description="summary.reason || '该实验未记录可用的类别系数矩阵，无法绘制系数热力图。'"
      />
    </div>

    <template v-else>
      <p v-if="classNotice" class="fallback-note">{{ classNotice }}</p>
      <p v-if="featureNotice" class="fallback-note">{{ featureNotice }}</p>

      <div ref="chartEl" class="chart"></div>
    </template>
  </el-card>
</template>

<style scoped>
.chart-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.chart-card :deep(.el-card__body) {
  padding: 6px 18px 16px;
}

.chart {
  width: 100%;
  height: 350px;
}

.chart-empty {
  display: grid;
  place-items: center;
  padding: 0 10px;
}

.chart-empty.compact {
  min-height: 180px;
}

@media (max-width: 720px) {
  .chart-card :deep(.el-card__body) {
    padding-right: 12px;
    padding-left: 12px;
  }
}
</style>
