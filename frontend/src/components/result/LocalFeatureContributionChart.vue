<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { formatNumber } from '@/utils/training'
import { contributionChartData } from '@/utils/modelExplanation'

const props = defineProps({
  sample: { type: Object, default: null },
  featureNames: { type: Array, default: () => [] },
  // Card title and the label of the reconstructed header tag differ per model:
  // regression -> 还原预测值, logistic -> 还原得分, SVM -> 还原得分.
  title: { type: String, default: '当前样本特征贡献' },
  subtitle: { type: String, default: '按贡献绝对值从大到小排列，正负贡献使用不同颜色' },
  reconstructedLabel: { type: String, default: '还原预测值' },
  predictedLabel: { type: String, default: '' },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const data = computed(() =>
  contributionChartData({ sample: props.sample, featureNames: props.featureNames }),
)

const hasData = computed(() => data.value.available)

const buildOption = () => {
  const rows = data.value.rows || []
  const showZoom = rows.length > 15

  return {
    animationDuration: 500,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const row = rows[params.dataIndex]
        if (!row) return ''
        return [
          `特征：${row.feature}`,
          `标准化后特征值：${formatNumber(row.featureValue)}`,
          `贡献值：${formatNumber(row.contribution)}`,
        ].join('<br/>')
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
        data: rows.map((row) => row.contribution),
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: (params) => {
            const row = rows[params.dataIndex]
            if (!row) return '#94a3b8'
            if (row.direction === 'positive') return '#059669'
            if (row.direction === 'negative') return '#dc2626'
            return '#94a3b8'
          },
        },
      },
    ],
  }
}

const { mount } = useChart(chartEl, buildOption, () => props.active && hasData.value)

useExportableChart({
  id: 'local-feature-contribution',
  section: '模型洞察',
  title: props.title,
  order: 6,
  available: () => hasData.value,
  buildOption,
})

watch(
  () => [props.sample, props.featureNames],
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
          <h3>{{ title }}</h3>
          <p>{{ subtitle }}</p>
        </div>
        <div class="header-tags">
          <el-tag v-if="predictedLabel" effect="plain" round type="success">
            {{ predictedLabel }}
          </el-tag>
          <el-tag effect="plain" round>截距 {{ formatNumber(data.intercept) }}</el-tag>
          <el-tag effect="plain" round>贡献合计 {{ formatNumber(data.contributionSum) }}</el-tag>
          <el-tag effect="plain" round>
            {{ reconstructedLabel }} {{ formatNumber(data.reconstructed) }}
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
