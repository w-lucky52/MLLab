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
  const points = diagnostics.value.residual_points
  const rows = points.map((point) => [point.predicted, point.residual, point.sample, point.actual])

  return {
    animationDuration: 900,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const [predicted, residual, sample, actual] = params.value
        return [
          `测试集样本序号：#${sample}`,
          `预测值：${Number(predicted).toFixed(2)}`,
          `真实值：${Number(actual).toFixed(2)}`,
          `残差（真实值 − 预测值）：${Number(residual).toFixed(2)}`,
        ].join('<br/>')
      },
    },
    grid: { top: 30, right: 30, bottom: 56, left: 64 },
    xAxis: {
      type: 'value',
      name: '预测值',
      nameLocation: 'middle',
      nameGap: 38,
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    yAxis: {
      type: 'value',
      name: '残差（真实值 − 预测值）',
      nameLocation: 'middle',
      nameGap: 54,
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    series: [
      {
        name: '残差',
        type: 'scatter',
        symbolSize: 10,
        data: rows,
        itemStyle: { color: '#7c3aed', opacity: 0.75 },
        markLine: {
          silent: true,
          symbol: 'none',
          label: { show: false },
          lineStyle: { color: '#94a3b8', type: 'dashed', width: 1 },
          data: [{ yAxis: 0 }],
        },
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
          <h3>预测值与残差</h3>
          <p>残差 = 真实值 − 预测值，正值表示预测偏低</p>
        </div>

        <el-tag v-if="hasData" effect="plain" round>残差散点</el-tag>
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
