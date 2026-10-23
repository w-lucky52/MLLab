<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yPred: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const toNumber = (value) => {
  const result = Number(value)
  return Number.isFinite(result) ? result : NaN
}

const points = computed(() => {
  const trueValues = Array.isArray(props.yTrue) ? props.yTrue : []
  const predictedValues = Array.isArray(props.yPred) ? props.yPred : []
  if (trueValues.length === 0 || trueValues.length !== predictedValues.length) return []

  const rows = []
  trueValues.forEach((value, index) => {
    const actual = toNumber(value)
    const predicted = toNumber(predictedValues[index])
    if (Number.isFinite(actual) && Number.isFinite(predicted)) {
      rows.push([actual, predicted])
    }
  })
  return rows
})

const hasData = computed(() => points.value.length > 0)

const buildOption = () => {
  const rows = points.value
  const values = rows.flat()
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)

  return {
    animationDuration: 900,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        return `真实值：${params.value[0].toFixed(2)}<br/>预测值：${params.value[1].toFixed(2)}`
      },
    },
    grid: { top: 30, right: 30, bottom: 60, left: 70 },
    xAxis: {
      type: 'value',
      name: '真实值',
      nameLocation: 'middle',
      nameGap: 38,
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    yAxis: {
      type: 'value',
      name: '预测值',
      nameLocation: 'middle',
      nameGap: 50,
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
  <div class="regression-performance">
    <el-card v-if="!hasData" class="chart-card" shadow="never">
      <div class="chart-empty">
        <el-empty :image-size="64" description="该实验未保存真实值与预测值，无法绘制散点图。" />
      </div>
    </el-card>

    <el-card v-else class="chart-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>真实值与预测值</h3>
            <p>预测结果与真实目标值的对应关系</p>
          </div>

          <el-tag effect="plain" round>回归结果</el-tag>
        </div>
      </template>

      <div ref="chartEl" class="chart"></div>
    </el-card>
  </div>
</template>

<style scoped>
.regression-performance {
  margin-top: 18px;
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

.chart-empty {
  display: grid;
  min-height: 380px;
  place-items: center;
}
</style>
