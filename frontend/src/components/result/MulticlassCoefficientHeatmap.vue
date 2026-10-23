<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { formatNumber } from '@/utils/training'
import {
  classNameNotice,
  explanationNotes,
  featureNameNotice,
  heatmapCells,
  interceptEntries,
  normalizeClassNames,
  normalizeFeatureNames,
  normalizeMatrix,
  symmetricRange,
} from '@/utils/modelExplanation'

const props = defineProps({
  coefficients: { type: [Array, Object], default: null },
  featureNames: { type: Array, default: null },
  classes: { type: Array, default: null },
  scoreClasses: { type: Array, default: null },
  intercept: { type: [Array, Number], default: null },
  modelName: { type: String, default: 'logistic_regression' },
  standardize: { type: Boolean, default: false },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const matrix = computed(() => normalizeMatrix(props.coefficients))
const cells = computed(() => heatmapCells(props.coefficients))
const range = computed(() => symmetricRange(props.coefficients))

const classNames = computed(() => {
  const rowCount = matrix.value?.rowCount ?? 0
  return normalizeClassNames(props.classes, props.scoreClasses, rowCount)
})

const featureNames = computed(() => {
  const columnCount = matrix.value?.columnCount ?? 0
  return normalizeFeatureNames(props.featureNames, columnCount)
})

const interceptList = computed(() =>
  interceptEntries(props.intercept, classNames.value),
)

const hasData = computed(() => Boolean(matrix.value && cells.value && range.value))
const classNotice = computed(() => classNameNotice(classNames.value))
const featureNotice = computed(() => featureNameNotice(featureNames.value))

const isLogistic = computed(() => props.modelName === 'logistic_regression')

const notes = computed(() =>
  explanationNotes('linear-classifier', { standardize: props.standardize }),
)

const cardTitle = computed(() => (isLogistic.value ? '类别 × 特征系数' : '类别 × 特征决策分数系数'))

const cardSubtitle = computed(() =>
  isLogistic.value
    ? '颜色以 0 为中心，展示各特征对对应类别 score（logit）的影响方向'
    : '颜色以 0 为中心，展示各特征对对应类别决策分数的影响方向',
)

const chartHeight = computed(() => {
  const count = matrix.value?.rowCount || 1
  return `${Math.min(560, Math.max(320, count * 46))}px`
})

const showZoom = computed(() => (matrix.value?.columnCount ?? 0) > 10)

const buildOption = () => {
  const data = cells.value
  const scale = range.value
  const rows = data?.cells || []
  const classLabels = classNames.value.names
  const featureLabels = featureNames.value.names
  const cellCount = (matrix.value?.rowCount ?? 0) * (matrix.value?.columnCount ?? 0)
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

watch(
  () => [props.coefficients, props.featureNames, props.classes, props.scoreClasses],
  () => {
    mount()
  },
  { deep: true },
)
</script>

<template>
  <el-card v-if="!hasData" class="chart-card" shadow="never">
    <div class="chart-empty">
      <el-empty
        :image-size="64"
        description="该实验未记录可用的类别系数矩阵，或矩阵形状不合法，无法绘制系数热力图。"
      />
    </div>
  </el-card>

  <template v-else>
    <div class="heatmap-layout">
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

        <p v-if="classNotice" class="fallback-note">{{ classNotice }}</p>
        <p v-if="featureNotice" class="fallback-note">{{ featureNotice }}</p>

        <div ref="chartEl" class="chart" :style="{ height: chartHeight }"></div>
      </el-card>

      <el-card class="intercept-card" shadow="never">
        <template #header>
          <div class="card-header">
            <div>
              <h3>各类别截距</h3>
              <p>行顺序与上方热力图一致</p>
            </div>
          </div>
        </template>

        <el-table
          v-if="interceptList"
          :data="interceptList"
          size="small"
          max-height="380"
          class="intercept-table"
        >
          <el-table-column prop="label" label="类别" min-width="90" show-overflow-tooltip />
          <el-table-column label="截距" min-width="100" align="right">
            <template #default="{ row }">{{ formatNumber(row.value) }}</template>
          </el-table-column>
        </el-table>

        <el-empty
          v-else
          :image-size="48"
          description="各类别截距未记录，或与类别数量不一致。"
        />
      </el-card>
    </div>

    <section class="insight-card">
      <span>系数说明</span>
      <ul>
        <li v-for="statement in notes" :key="statement">{{ statement }}</li>
      </ul>
    </section>
  </template>
</template>

<style scoped>
.heatmap-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.9fr) minmax(0, 1fr);
  gap: 18px;
}

.chart-card :deep(.el-card__header),
.intercept-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.chart-card :deep(.el-card__body) {
  padding: 6px 18px 16px;
}

.intercept-card :deep(.el-card__body) {
  padding: 8px 14px 16px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
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

.fallback-note {
  margin: 8px 4px 0;
  color: #b45309;
  font-size: 11px;
  line-height: 1.6;
}

.chart {
  width: 100%;
}

.chart-empty {
  display: grid;
  min-height: 320px;
  place-items: center;
  padding: 0 10px;
}

.chart-empty :deep(.el-empty__description),
.intercept-card :deep(.el-empty__description) {
  max-width: 460px;
}

.chart-empty :deep(.el-empty__description p),
.intercept-card :deep(.el-empty__description p) {
  color: #64748b;
  font-size: 12px;
  line-height: 1.8;
}

.intercept-table :deep(.el-table__cell) {
  font-size: 12px;
}

.insight-card {
  padding: 20px 22px;
  border: 1px solid #dbeafe;
  border-radius: 17px;
  background: linear-gradient(135deg, #eff6ff, #f8faff);
}

.insight-card > span {
  color: #3b82f6;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.2px;
}

.insight-card ul {
  padding: 0;
  margin: 10px 0 0;
  list-style: none;
}

.insight-card li {
  position: relative;
  padding-left: 16px;
  margin-bottom: 7px;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.insight-card li:last-child {
  margin-bottom: 0;
}

.insight-card li::before {
  position: absolute;
  top: 8px;
  left: 0;
  width: 5px;
  height: 5px;
  content: '';
  border-radius: 50%;
  background: #3b82f6;
}

@media (max-width: 1000px) {
  .heatmap-layout {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
