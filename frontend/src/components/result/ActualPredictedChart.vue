<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { regressionDiagnostics } from '@/utils/resultDiagnostics'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yPred: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const diagnostics = computed(() => regressionDiagnostics(props.yTrue, props.yPred))
const hasData = computed(() => diagnostics.value.available)

const buildOption = () => {
  const points = diagnostics.value.actual_predicted_points
  const rows = points.map((point) => [point.actual, point.predicted, point.sample])
  const values = rows.flatMap((row) => [row[0], row[1]])
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)

  return {
    animationDuration: 900,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const [actual, predicted, sample] = params.value
        return [
          `测试集样本序号：#${sample}`,
          `真实值：${Number(actual).toFixed(2)}`,
          `预测值：${Number(predicted).toFixed(2)}`,
        ].join('<br/>')
      },
    },
    grid: { top: 30, right: 30, bottom: 56, left: 64 },
    xAxis: {
      type: 'value',
      name: '真实值',
      nameLocation: 'middle',
      nameGap: 38,
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    yAxis: {
      type: 'value',
      name: '预测值',
      nameLocation: 'middle',
      nameGap: 48,
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    series: [
      {
        name: '预测样本',
        type: 'scatter',
        symbolSize: 10,
        data: rows,
        itemStyle: { color: '#3b82f6', opacity: 0.75 },
      },
      {
        name: '理想预测线',
        type: 'line',
        data: [
          [minimum, minimum],
          [maximum, maximum],
        ],
        symbol: 'none',
        silent: true,
        tooltip: { show: false },
        lineStyle: { color: '#ef4444', type: 'dashed', width: 2 },
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
  () => [props.yTrue, props.yPred],
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
          <h3>真实值与预测值</h3>
          <p>预测结果与真实目标值的对应关系</p>
        </div>

        <el-tag v-if="hasData" effect="plain" round>回归结果</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty compact">
      <el-empty :image-size="56" :description="diagnostics.reason" />
    </div>

    <div v-else ref="chartEl" class="chart"></div>
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
  height: 320px;
}

.chart-empty {
  display: grid;
  min-height: 320px;
  place-items: center;
}

.chart-empty.compact {
  min-height: 132px;
}
</style>
