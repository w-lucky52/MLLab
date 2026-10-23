<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { formatNumber } from '@/utils/training'
import {
  classNameNotice,
  explanationNotes,
  featureNameNotice,
  gaussianExplanationData,
  heatmapCells,
  nonNegativeRange,
  symmetricRange,
} from '@/utils/modelExplanation'

const props = defineProps({
  trainingInfo: { type: Object, default: null },
  featureNames: { type: Array, default: null },
  scoreClasses: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)
const mode = ref('mean')

const data = computed(() =>
  gaussianExplanationData({
    trainingInfo: props.trainingInfo,
    featureNames: props.featureNames,
    scoreClasses: props.scoreClasses,
  }),
)

const matrix = computed(() => {
  if (!data.value) return null
  return mode.value === 'variance' ? data.value.variances : data.value.means
})

const cells = computed(() => heatmapCells(matrix.value))
const range = computed(() => {
  if (!matrix.value) return null
  return mode.value === 'variance'
    ? nonNegativeRange(matrix.value)
    : symmetricRange(matrix.value)
})

const hasData = computed(() => Boolean(data.value && cells.value && range.value))
const classNotice = computed(() => classNameNotice(data.value?.classes))
const featureNotice = computed(() => featureNameNotice(data.value?.featureNames))
const notes = computed(() => explanationNotes('gaussian-nb', props.trainingInfo))
const standardizeNote = computed(() =>
  props.trainingInfo?.standardize === true
    ? '本次训练特征已标准化，统计量基于标准化后的训练特征。'
    : '',
)

const statisticName = computed(() => (mode.value === 'variance' ? '方差' : '均值'))

const cards = computed(() => {
  if (!data.value) return []
  return [
    { key: 'classes', label: '类别数量', value: String(data.value.rowCount) },
    { key: 'features', label: '特征数量', value: String(data.value.columnCount) },
    {
      key: 'scale',
      label: '当前尺度',
      value: mode.value === 'variance' ? '类内方差' : '类内均值',
      hint:
        mode.value === 'variance'
          ? `颜色从 0 到 ${formatNumber(range.value?.maximum ?? 0)}`
          : '颜色以 0 为中心',
    },
  ]
})

const chartHeight = computed(() => {
  const count = data.value?.rowCount || 1
  return `${Math.min(560, Math.max(320, count * 46))}px`
})

const showZoom = computed(() => (data.value?.columnCount ?? 0) > 10)

const buildOption = () => {
  const cellData = cells.value
  const scale = range.value
  const current = data.value
  const label = statisticName.value

  return {
    animationDuration: 400,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const [columnIndex, rowIndex, value] = params.value
        return `类别：${current.classes.names[rowIndex] ?? ''}<br/>特征：${current.featureNames.names[columnIndex] ?? ''}<br/>${label}：${formatNumber(value)}`
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
      data: current.featureNames.names,
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
      axisLabel: {
        color: '#64748b',
        fontSize: 11,
        interval: 0,
        rotate: current.featureNames.names.length > 8 ? 35 : 0,
        width: 96,
        overflow: 'truncate',
      },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: current.classes.names,
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
      axisLabel: { color: '#475569', fontSize: 11, width: 92, overflow: 'truncate' },
    },
    visualMap: {
      min: scale?.min ?? 0,
      max: scale?.max ?? 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 4,
      itemWidth: 12,
      itemHeight: 110,
      textStyle: { color: '#94a3b8', fontSize: 10 },
      inRange: {
        color:
          mode.value === 'variance'
            ? ['#f0fdf4', '#bbf7d0', '#34d399', '#047857']
            : ['#dc2626', '#f8fafc', '#059669'],
      },
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
        data: cellData?.cells || [],
        label: {
          show: (cellData?.rowCount ?? 0) * (cellData?.columnCount ?? 0) <= 24,
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

watch(mode, () => {
  mount()
})

watch(
  () => [props.trainingInfo, props.featureNames, props.scoreClasses],
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
        description="该实验未记录完整的高斯朴素贝叶斯均值与方差矩阵，或矩阵形状不一致，无法绘制统计热力图。"
      />
    </div>
  </el-card>

  <template v-else>
    <el-card class="chart-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>类别 × 特征统计量</h3>
            <p>行列顺序严格使用后端记录的顺序</p>
          </div>

          <el-radio-group v-model="mode" size="small">
            <el-radio-button value="mean">均值</el-radio-button>
            <el-radio-button value="variance">方差</el-radio-button>
          </el-radio-group>
        </div>
      </template>

      <section class="summary-grid">
        <div v-for="card in cards" :key="card.key" class="summary-item">
          <span>{{ card.label }}</span>
          <strong>{{ card.value }}</strong>
          <p v-if="card.hint">{{ card.hint }}</p>
        </div>
      </section>

      <p v-if="standardizeNote" class="standardize-note">{{ standardizeNote }}</p>
      <p v-if="classNotice" class="fallback-note">{{ classNotice }}</p>
      <p v-if="featureNotice" class="fallback-note">{{ featureNotice }}</p>

      <div ref="chartEl" class="chart" :style="{ height: chartHeight }"></div>
    </el-card>

    <section class="insight-card">
      <span>统计量说明</span>
      <ul>
        <li v-for="statement in notes" :key="statement">{{ statement }}</li>
      </ul>
    </section>
  </template>
</template>

<style scoped>
.chart-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.chart-card :deep(.el-card__body) {
  padding: 6px 18px 16px;
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

.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  padding: 12px 4px 4px;
}

.summary-item {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 5px;
  padding: 11px 14px;
  border-radius: 12px;
  background: #f8fafc;
}

.summary-item span {
  color: #94a3b8;
  font-size: 11px;
}

.summary-item strong {
  overflow: hidden;
  color: #0f172a;
  font-size: 15px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-item p {
  margin: 0;
  color: #94a3b8;
  font-size: 10px;
}

.standardize-note,
.fallback-note {
  margin: 10px 4px 0;
  font-size: 11px;
  line-height: 1.6;
}

.standardize-note {
  color: #2563eb;
}

.fallback-note {
  color: #b45309;
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

.chart-empty :deep(.el-empty__description) {
  max-width: 460px;
}

.chart-empty :deep(.el-empty__description p) {
  color: #64748b;
  font-size: 12px;
  line-height: 1.8;
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

@media (max-width: 700px) {
  .summary-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
