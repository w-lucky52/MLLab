<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { regressionDiagnostics } from '@/utils/resultDiagnostics'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yPred: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const diagnostics = computed(() => regressionDiagnostics(props.yTrue, props.yPred))
const rows = computed(() =>
  diagnostics.value.available ? diagnostics.value.top_absolute_errors : [],
)
const hasData = computed(() => rows.value.length > 0)

const buildOption = () => {
  const items = rows.value
  const labels = items.map((item) => `测试样本 #${item.sample}`)

  return {
    animationDuration: 700,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const item = items[params.dataIndex]
        if (!item) return ''
        return [
          `测试集样本序号：#${item.sample}`,
          `真实值：${Number(item.actual).toFixed(2)}`,
          `预测值：${Number(item.predicted).toFixed(2)}`,
          `残差：${Number(item.residual).toFixed(2)}`,
          `绝对误差：${Number(item.absoluteError).toFixed(2)}`,
        ].join('<br/>')
      },
    },
    grid: { top: 24, right: 34, bottom: 40, left: 98 },
    xAxis: {
      type: 'value',
      name: '绝对误差',
      nameLocation: 'middle',
      nameGap: 26,
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    yAxis: {
      type: 'category',
      data: labels,
      inverse: true,
      axisLabel: { color: '#64748b', fontSize: 11 },
      axisTick: { show: false },
    },
    series: [
      {
        name: '绝对误差',
        type: 'bar',
        barMaxWidth: 18,
        data: items.map((item) => item.absoluteError),
        itemStyle: { color: '#ea580c', borderRadius: [0, 3, 3, 0] },
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
  id: 'top-absolute-errors',
  section: '回归表现',
  title: '绝对误差最大的样本',
  order: 4,
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
          <h3>绝对误差最大的样本</h3>
          <p>按绝对误差降序排列，最多展示 10 个测试样本</p>
        </div>

        <el-tag v-if="hasData" effect="plain" round>前 {{ rows.length }} 个</el-tag>
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
