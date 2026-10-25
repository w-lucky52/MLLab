<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { formatNumber } from '@/utils/training'
import { featureNameNotice, summarizeCoefficients } from '@/utils/modelExplanation'

const props = defineProps({
  coefficients: { type: [Array, Object], default: null },
  featureNames: { type: Array, default: null },
  intercept: { type: [Array, Number], default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const summary = computed(() =>
  summarizeCoefficients({
    coefficients: props.coefficients,
    featureNames: props.featureNames,
    intercept: props.intercept,
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
        const direction =
          row.direction === 'positive'
            ? '正（预测值倾向增加）'
            : row.direction === 'negative'
              ? '负（预测值倾向降低）'
              : '零（不改变预测值）'
        return `特征：${row.feature}<br/>系数原始值：${formatNumber(row.value)}<br/>方向：${direction}`
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
      axisLabel: {
        color: '#475569',
        fontSize: 11,
        width: 130,
        overflow: 'truncate',
      },
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

const { mount } = useChart(
  chartEl,
  buildOption,
  () => props.active && hasData.value,
)

useExportableChart({
  id: 'coefficient',
  section: '模型洞察',
  title: '特征系数',
  order: 1,
  available: () => hasData.value,
  buildOption,
})

watch(
  () => [props.coefficients, props.featureNames, props.intercept],
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
          <h3>特征系数</h3>
          <p>按系数绝对值从大到小排列，正负系数使用不同颜色，以 0 为基准</p>
        </div>
        <div class="header-tags">
          <el-tag v-if="summary.intercept !== null && summary.intercept !== undefined" effect="plain" round>
            截距 {{ formatNumber(summary.intercept) }}
          </el-tag>
          <el-tag effect="plain" round>线性回归</el-tag>
        </div>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty compact">
      <el-empty
        :image-size="56"
        :description="summary.reason || '该实验未记录可用的特征系数，无法绘制系数图。'"
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
