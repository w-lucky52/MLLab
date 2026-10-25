<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { formatNumber } from '@/utils/training'
import { classDisplayLookup, classDisplayName } from '@/utils/classDisplay'
import {
  classNameNotice,
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
  scoreClassNames: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const meanChartEl = ref(null)
const varianceChartEl = ref(null)

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const data = computed(() =>
  gaussianExplanationData({
    trainingInfo: props.trainingInfo,
    featureNames: props.featureNames,
    scoreClasses: props.scoreClasses,
  }),
)

const hasData = computed(() => data.value !== null)

// Row order stays untouched; only the class labels change for display.
const displayClassNames = computed(() =>
  (data.value?.classes.names ?? []).map((name) => classDisplayName(name, displayLookup.value)),
)

const classNotice = computed(() => classNameNotice(data.value?.classes))
const featureNotice = computed(() => featureNameNotice(data.value?.featureNames))

const buildOptionFor = (kind) => {
  const current = data.value
  const classLabels = displayClassNames.value
  const featureLabels = current.featureNames.names
  const matrix = kind === 'variance' ? current.variances : current.means
  const cells = heatmapCells(matrix)
  const scale = kind === 'variance' ? nonNegativeRange(matrix) : symmetricRange(matrix)
  const statisticLabel = kind === 'variance' ? '方差' : '均值'
  const cellCount = current.rowCount * current.columnCount
  const showLabels = cellCount <= 24
  const zoom = current.columnCount > 10

  return {
    animationDuration: 400,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const [columnIndex, rowIndex, value] = params.value
        return `类别：${classLabels[rowIndex] ?? ''}<br/>特征：${featureLabels[columnIndex] ?? ''}<br/>${statisticLabel}：${formatNumber(value)}`
      },
    },
    grid: {
      top: 16,
      right: 20,
      bottom: zoom ? 92 : 64,
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
        width: 80,
        overflow: 'truncate',
      },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: classLabels,
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
      axisLabel: { color: '#475569', fontSize: 11, width: 80, overflow: 'truncate' },
    },
    visualMap: {
      min: scale?.min ?? 0,
      max: scale?.max ?? 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 4,
      itemWidth: 12,
      itemHeight: 100,
      textStyle: { color: '#94a3b8', fontSize: 10 },
      inRange: {
        color:
          kind === 'variance'
            ? ['#f0fdf4', '#bbf7d0', '#34d399', '#047857']
            : ['#dc2626', '#f8fafc', '#059669'],
      },
    },
    dataZoom: zoom
      ? [
          { type: 'inside', xAxisIndex: 0 },
          { type: 'slider', xAxisIndex: 0, height: 14, bottom: 34 },
        ]
      : [],
    series: [
      {
        type: 'heatmap',
        data: cells?.cells || [],
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

const buildMeanOption = () => buildOptionFor('mean')
const buildVarianceOption = () => buildOptionFor('variance')

const isChartActive = () => props.active && hasData.value
const meanChart = useChart(meanChartEl, buildMeanOption, isChartActive)
const varianceChart = useChart(varianceChartEl, buildVarianceOption, isChartActive)

useExportableChart({
  id: 'gnb-means',
  section: '模型洞察',
  title: '类别 × 特征均值',
  order: 4,
  available: () => hasData.value,
  buildOption: buildMeanOption,
})

useExportableChart({
  id: 'gnb-variances',
  section: '模型洞察',
  title: '类别 × 特征方差',
  order: 5,
  available: () => hasData.value,
  buildOption: buildVarianceOption,
})

watch(
  () => [props.trainingInfo, props.featureNames, props.scoreClasses, props.scoreClassNames],
  () => {
    meanChart.mount()
    varianceChart.mount()
  },
  { deep: true },
)
</script>

<template>
  <el-card v-if="!hasData" class="chart-card" shadow="never">
    <div class="chart-empty compact">
      <el-empty
        :image-size="56"
        description="该实验未记录完整且形状一致的类别均值与方差矩阵，无法绘制统计热力图。"
      />
    </div>
  </el-card>

  <div v-else class="heatmap-pair">
    <el-card class="chart-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>类别 × 特征均值</h3>
            <p>各类别训练样本的特征均值，颜色以 0 为中心</p>
          </div>
          <el-tag effect="plain" round>均值</el-tag>
        </div>
      </template>

      <p v-if="classNotice" class="fallback-note">{{ classNotice }}</p>
      <p v-if="featureNotice" class="fallback-note">{{ featureNotice }}</p>

      <div ref="meanChartEl" class="chart"></div>
    </el-card>

    <el-card class="chart-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>类别 × 特征方差</h3>
            <p>各类别训练样本的特征方差，颜色从 0 开始</p>
          </div>
          <el-tag effect="plain" round>方差</el-tag>
        </div>
      </template>

      <div ref="varianceChartEl" class="chart"></div>
    </el-card>
  </div>
</template>

<style scoped>
.heatmap-pair {
  display: grid;
  align-items: stretch;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.heatmap-pair > * {
  min-width: 0;
  height: 100%;
}

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

@media (max-width: 1100px) {
  .heatmap-pair {
    grid-template-columns: minmax(0, 1fr);
  }

  .heatmap-pair > * {
    height: auto;
  }
}

@media (max-width: 720px) {
  .chart-card :deep(.el-card__body) {
    padding-right: 12px;
    padding-left: 12px;
  }
}
</style>
