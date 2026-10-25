<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { formatNumber } from '@/utils/training'
import { permutationImportanceData } from '@/utils/modelExplanation'

const props = defineProps({
  // Raw ``training_info.explanation_data`` payload (kind == permutation_importance).
  explanation: { type: Object, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const data = computed(() => permutationImportanceData(props.explanation))
const hasData = computed(() => data.value.available)

const buildOption = () => {
  const rows = data.value.rows || []
  const deltaLabel = data.value.deltaLabel
  const showZoom = rows.length > 15

  return {
    animationDuration: 500,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const row = rows[params.dataIndex]
        if (!row) return ''
        return `特征：${row.feature}<br/>${deltaLabel}：${formatNumber(row.importance)}`
      },
    },
    grid: { top: 16, right: 34, bottom: showZoom ? 46 : 16, left: 12, containLabel: true },
    xAxis: {
      type: 'value',
      name: deltaLabel,
      nameLocation: 'middle',
      nameGap: 26,
      nameTextStyle: { color: '#94a3b8', fontSize: 11 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: '#94a3b8', fontSize: 11 },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: rows.map((row) => row.feature),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: '#475569', fontSize: 11, width: 130, overflow: 'truncate' },
    },
    dataZoom: showZoom
      ? [
          { type: 'inside', yAxisIndex: 0 },
          { type: 'slider', yAxisIndex: 0, width: 10, right: 6, bottom: 16, top: 16 },
        ]
      : [],
    series: [
      {
        type: 'bar',
        barMaxWidth: 18,
        data: rows.map((row) => row.importance),
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: (params) => {
            const row = rows[params.dataIndex]
            if (!row) return '#94a3b8'
            if (row.direction === 'positive') return '#2563eb'
            if (row.direction === 'negative') return '#f59e0b'
            return '#94a3b8'
          },
        },
      },
    ],
  }
}

const { mount } = useChart(chartEl, buildOption, () => props.active && hasData.value)

useExportableChart({
  id: 'permutation-importance',
  section: '模型洞察',
  title: '测试集置换重要性',
  order: 10,
  available: () => hasData.value,
  buildOption,
})

watch(
  () => props.explanation,
  () => {
    mount()
  },
  { deep: true },
)
</script>

<template>
  <el-card v-if="hasData" class="chart-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>测试集置换重要性</h3>
          <p>每次只置换一个特征列，按{{ data.deltaLabel }}从大到小排列</p>
        </div>
        <div class="header-tags">
          <el-tag effect="plain" round>{{ data.metricTag }}</el-tag>
          <el-tag effect="plain" round>
            {{ data.baselineLabel }} {{ formatNumber(data.baseline) }}
          </el-tag>
        </div>
      </div>
    </template>

    <div ref="chartEl" class="chart"></div>
  </el-card>
</template>

<style scoped>
.chart-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.chart-card :deep(.el-card__body) {
  padding: 6px 18px 16px;
}

.header-tags {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.chart {
  width: 100%;
  height: 350px;
}

@media (max-width: 720px) {
  .chart-card :deep(.el-card__header) {
    padding: 16px 14px;
  }

  .chart-card :deep(.el-card__body) {
    padding-right: 12px;
    padding-left: 12px;
  }
}
</style>
