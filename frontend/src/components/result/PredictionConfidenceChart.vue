<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { confidenceHistogram } from '@/utils/resultDiagnostics'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yPred: { type: Array, default: null },
  yScore: { type: Array, default: null },
  scoreType: { type: String, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const histogram = computed(() =>
  confidenceHistogram(props.yTrue, props.yPred, props.yScore, props.scoreType, 10),
)

const hasData = computed(
  () => histogram.value.available && histogram.value.total > 0 && histogram.value.bins.length > 0,
)

const formatEdge = (value) => {
  if (!Number.isFinite(value)) return '—'
  return String(Number(value.toFixed(3)))
}

const binLabel = (bin) => {
  if (histogram.value.bins.length === 1) return `全部为 ${formatEdge(bin.min)}`
  return `${formatEdge(bin.min)} ~ ${formatEdge(bin.max)}`
}

const buildOption = () => {
  const bins = histogram.value.bins
  const labels = bins.map(binLabel)

  return {
    animationDuration: 700,
    color: ['#059669', '#ef4444'],
    legend: {
      top: 0,
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { color: '#475569', fontSize: 12 },
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const first = Array.isArray(params) ? params[0] : params
        const index = first?.dataIndex ?? 0
        const bin = bins[index]
        if (!bin) return ''
        return [
          `区间：${labels[index]}`,
          `预测正确：${bin.correct}`,
          `预测错误：${bin.wrong}`,
        ].join('<br/>')
      },
    },
    grid: { top: 36, right: 24, bottom: 52, left: 46 },
    xAxis: {
      type: 'category',
      data: labels,
      axisTick: { alignWithLabel: true },
      axisLabel: {
        color: '#64748b',
        fontSize: 10,
        interval: 0,
        rotate: bins.length > 6 ? 30 : 0,
      },
    },
    yAxis: {
      type: 'value',
      name: '样本数',
      minInterval: 1,
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    series: [
      {
        name: '预测正确',
        type: 'bar',
        stack: 'samples',
        barMaxWidth: 26,
        data: bins.map((bin) => bin.correct),
      },
      {
        name: '预测错误',
        type: 'bar',
        stack: 'samples',
        barMaxWidth: 26,
        data: bins.map((bin) => bin.wrong),
        itemStyle: { borderRadius: [3, 3, 0, 0] },
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
  () => [props.yTrue, props.yPred, props.yScore, props.scoreType],
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
          <h3>{{ hasData ? histogram.title : '预测置信度分布' }}</h3>
          <p>{{ hasData ? histogram.description : '根据预测分数与真实标签统计' }}</p>
        </div>

        <el-tag v-if="hasData" effect="plain" round>
          {{ histogram.kind === 'decision' ? '决策分数（非概率）' : '预测概率' }}
        </el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty compact">
      <el-empty :image-size="56" :description="histogram.reason || '该实验未保存预测分数。'" />
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
