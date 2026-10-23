<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'

const props = defineProps({
  confusionMatrix: { type: Array, default: null },
  classLabels: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)
const mode = ref('count')

const toNumber = (value) => {
  const result = Number(value)
  return Number.isFinite(result) ? result : 0
}

const matrix = computed(() => {
  if (!Array.isArray(props.confusionMatrix) || props.confusionMatrix.length === 0) return []
  return props.confusionMatrix.map((row) => (Array.isArray(row) ? row.map(toNumber) : []))
})

const hasData = computed(() => matrix.value.length > 0)

const labels = computed(() => {
  const source = Array.isArray(props.classLabels) ? props.classLabels.map(String) : []
  if (source.length === matrix.value.length) return source
  return matrix.value.map((_, index) => `类别 ${index + 1}`)
})

// Row-normalised percentages (rows are true classes); zero-sum rows stay 0.
const percentMatrix = computed(() =>
  matrix.value.map((row) => {
    const total = row.reduce((sum, value) => sum + value, 0)
    return row.map((value) => (total > 0 ? (value / total) * 100 : 0))
  }),
)

const buildOption = () => {
  const usePercent = mode.value === 'percent'
  const values = usePercent ? percentMatrix.value : matrix.value
  const axisLabels = labels.value
  const maxCount = Math.max(1, ...matrix.value.flat())

  const data = values.flatMap((row, rowIndex) =>
    row.map((value, columnIndex) => [columnIndex, rowIndex, Number(value.toFixed(2))]),
  )

  return {
    animationDuration: 700,
    tooltip: {
      position: 'top',
      formatter: (params) => {
        const [columnIndex, rowIndex] = params.value
        const count = matrix.value[rowIndex][columnIndex]
        const percent = percentMatrix.value[rowIndex][columnIndex]
        const lines = [
          `真实类别：${axisLabels[rowIndex]}`,
          `预测类别：${axisLabels[columnIndex]}`,
          `样本数量：${count}`,
        ]
        if (usePercent) lines.push(`占真实类别比例：${percent.toFixed(2)}%`)
        return lines.join('<br/>')
      },
    },
    grid: {
      top: 24,
      right: 24,
      bottom: axisLabels.length > 6 ? 82 : 62,
      left: axisLabels.length > 6 ? 86 : 74,
    },
    xAxis: {
      type: 'category',
      data: axisLabels,
      name: '预测类别',
      nameLocation: 'middle',
      nameGap: axisLabels.length > 6 ? 68 : 44,
      splitArea: { show: true },
      axisLabel: {
        color: '#64748b',
        interval: 0,
        fontSize: 11,
        rotate: axisLabels.length > 6 ? 35 : 0,
      },
    },
    yAxis: {
      type: 'category',
      data: axisLabels,
      inverse: true,
      name: '真实类别',
      nameLocation: 'middle',
      nameGap: axisLabels.length > 6 ? 62 : 50,
      splitArea: { show: true },
      axisLabel: {
        color: '#64748b',
        interval: 0,
        fontSize: 11,
      },
    },
    visualMap: {
      min: 0,
      max: usePercent ? 100 : maxCount,
      show: false,
      inRange: { color: ['#eff6ff', '#93c5fd', '#2563eb'] },
    },
    series: [
      {
        name: usePercent ? '混淆矩阵（百分比）' : '混淆矩阵（数量）',
        type: 'heatmap',
        data,
        label: {
          show: true,
          color: '#0f172a',
          fontWeight: 700,
          fontSize: axisLabels.length > 6 ? 10 : 12,
          formatter: (params) =>
            usePercent ? `${params.value[2].toFixed(1)}%` : `${params.value[2]}`,
        },
      },
    ],
  }
}

const { mount, render } = useChart(
  chartEl,
  buildOption,
  () => props.active && hasData.value,
)

watch(mode, () => {
  render()
})

watch(
  () => [props.confusionMatrix, props.classLabels],
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
          <h3>混淆矩阵</h3>
          <p>真实类别与预测类别的对应关系</p>
        </div>

        <el-radio-group v-model="mode" size="small">
          <el-radio-button value="count">数量</el-radio-button>
          <el-radio-button value="percent">百分比</el-radio-button>
        </el-radio-group>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty">
      <el-empty :image-size="64" description="该实验未保存混淆矩阵数据。" />
    </div>

    <div v-else ref="chartEl" class="chart" :class="{ dense: labels.length > 6 }"></div>
  </el-card>
</template>

<style scoped>
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

.chart {
  width: 100%;
  height: 380px;
}

.chart.dense {
  height: 440px;
}

.chart-empty {
  display: grid;
  min-height: 380px;
  place-items: center;
}
</style>
