<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { formatNumber } from '@/utils/training'
import {
  featureNameNotice,
  summarizeFeatureImportances,
} from '@/utils/modelExplanation'

const props = defineProps({
  featureImportances: { type: [Array, Object], default: null },
  featureNames: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const summary = computed(() =>
  summarizeFeatureImportances({
    featureImportances: props.featureImportances,
    featureNames: props.featureNames,
  }),
)

const hasData = computed(() => summary.value.available)
const featureNotice = computed(() => featureNameNotice(summary.value.featureNames))

const buildOption = () => {
  const rows = summary.value.rows || []
  const showZoom = rows.length > 15

  return {
    animationDuration: 500,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const row = rows[params.dataIndex]
        if (!row) return ''
        const percent =
          row.percent === null ? '' : `<br/>占比：${formatNumber(row.percent)}%（基于重要性总和）`
        return `特征：${row.feature}<br/>原始重要性：${formatNumber(row.value)}${percent}`
      },
    },
    grid: { top: 16, right: 34, bottom: showZoom ? 46 : 16, left: 12, containLabel: true },
    xAxis: {
      type: 'value',
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
        data: rows.map((row) => row.value),
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: '#2563eb',
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

useExportableChart({
  id: 'feature-importance',
  section: '模型洞察',
  title: '特征重要性',
  order: 3,
  available: () => hasData.value,
  buildOption,
})

watch(
  () => [props.featureImportances, props.featureNames],
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
          <h3>特征重要性</h3>
          <p>使用训练记录中的原始重要性，按数值从大到小排列</p>
        </div>
        <el-tag effect="plain" round>树模型</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty compact">
      <el-empty
        :image-size="56"
        :description="summary.reason || '该实验未记录可用的特征重要性，无法生成重要性图。'"
      />
    </div>

    <template v-else>
      <p v-if="featureNotice" class="fallback-note">{{ featureNotice }}</p>

      <div ref="chartEl" class="chart"></div>
    </template>
  </el-card>
</template>

<style scoped>
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

@media (max-width: 720px) {
  .chart-card :deep(.el-card__body) {
    padding-right: 12px;
    padding-left: 12px;
  }
}
</style>
