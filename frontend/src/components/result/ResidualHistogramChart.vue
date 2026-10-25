<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { buildBins, regressionDiagnostics } from '@/utils/resultDiagnostics'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yPred: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const diagnostics = computed(() => regressionDiagnostics(props.yTrue, props.yPred))
const histogram = computed(() =>
  diagnostics.value.available ? buildBins(diagnostics.value.residuals, 10) : { bins: [] },
)
const hasData = computed(
  () => diagnostics.value.available && histogram.value.available && histogram.value.bins.length > 0,
)

const formatEdge = (value) => {
  if (!Number.isFinite(value)) return '—'
  return String(Number(value.toFixed(3)))
}

const binLabel = (bin) => {
  if (histogram.value.bins.length === 1) return `全部为 ${formatEdge(bin.min)}`
  return `${formatEdge(bin.min)} ~ ${formatEdge(bin.max)}`
}

const zeroBinIndex = computed(() => {
  const bins = histogram.value.bins
  return bins.findIndex((bin, index) => {
    if (index === bins.length - 1) return bin.min <= 0 && 0 <= bin.max
    return bin.min <= 0 && 0 < bin.max
  })
})

const buildOption = () => {
  const bins = histogram.value.bins
  const labels = bins.map(binLabel)
  const zeroIndex = zeroBinIndex.value

  const markLine =
    zeroIndex >= 0
      ? {
          silent: true,
          symbol: 'none',
          label: { show: true, formatter: '残差 0', color: '#94a3b8', fontSize: 10 },
          lineStyle: { color: '#94a3b8', type: 'dashed', width: 1 },
          data: [{ xAxis: labels[zeroIndex] }],
        }
      : undefined

  return {
    animationDuration: 700,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const first = Array.isArray(params) ? params[0] : params
        const index = first?.dataIndex ?? 0
        const bin = bins[index]
        if (!bin) return ''
        return [`残差区间：${labels[index]}`, `样本数：${bin.count}`].join('<br/>')
      },
    },
    grid: { top: 30, right: 24, bottom: 56, left: 46 },
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
        name: '样本数',
        type: 'bar',
        barMaxWidth: 26,
        data: bins.map((bin) => bin.count),
        itemStyle: { color: '#0891b2', borderRadius: [3, 3, 0, 0] },
        markLine,
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
  id: 'residual-histogram',
  section: '回归表现',
  title: '残差分布',
  order: 3,
  available: () => hasData.value,
  buildOption,
})

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
          <h3>残差分布</h3>
          <p>残差越集中在 0 附近，表示预测误差整体越小</p>
        </div>

        <el-tag v-if="hasData" effect="plain" round>10 个区间</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty compact">
      <el-empty :image-size="56" :description="diagnostics.reason" />
    </div>

    <div v-else ref="chartEl" class="chart"></div>

    <p v-if="hasData" class="chart-note">
      残差越集中在 0 附近，表示预测误差整体越小；分布偏向某一侧可能表示模型存在系统性高估或低估，但不能仅凭该图得出因果结论。
    </p>
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

.chart-note {
  margin: 8px 2px 0;
  color: #94a3b8;
  font-size: 11px;
  line-height: 1.6;
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
