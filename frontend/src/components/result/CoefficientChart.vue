<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { formatNumber } from '@/utils/training'
import {
  explanationNotes,
  featureNameNotice,
  regressionCoefficientSummary,
} from '@/utils/modelExplanation'

const props = defineProps({
  coefficients: { type: [Array, Object], default: null },
  featureNames: { type: Array, default: null },
  intercept: { type: [Array, Number], default: null },
  standardize: { type: Boolean, default: false },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const summary = computed(() =>
  regressionCoefficientSummary({
    coefficients: props.coefficients,
    featureNames: props.featureNames,
    intercept: props.intercept,
  }),
)

const hasData = computed(() => Boolean(summary.value))
const featureNotice = computed(() => featureNameNotice(summary.value?.featureNames))
const notes = computed(() =>
  explanationNotes('linear-regression', { standardize: props.standardize }),
)

const cards = computed(() => {
  const data = summary.value
  if (!data) return []

  return [
    {
      key: 'intercept',
      label: '截距',
      value: data.intercept === null ? '未记录' : formatNumber(data.intercept),
    },
    { key: 'features', label: '特征数量', value: String(data.featureCount) },
    {
      key: 'top',
      label: '最大绝对系数特征',
      value: data.topFeature ?? '未记录',
      hint:
        data.topMagnitude === null ? '' : `|系数| = ${formatNumber(data.topMagnitude)}`,
    },
  ]
})

const chartHeight = computed(() => {
  const count = summary.value?.rows.length || 0
  return `${Math.min(620, Math.max(300, count * 32))}px`
})

const buildOption = () => {
  const data = summary.value
  const rows = data?.rows || []
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
        <el-tag effect="plain" round>线性回归</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty">
      <el-empty
        :image-size="64"
        description="该实验未记录可用的特征系数，或系数数据不合法，无法绘制系数图。"
      />
    </div>

    <template v-else>
      <section class="summary-grid">
        <div v-for="card in cards" :key="card.key" class="summary-item">
          <span>{{ card.label }}</span>
          <strong>{{ card.value }}</strong>
          <p v-if="card.hint">{{ card.hint }}</p>
        </div>
      </section>

      <p v-if="featureNotice" class="fallback-note">{{ featureNotice }}</p>

      <div ref="chartEl" class="chart" :style="{ height: chartHeight }"></div>
    </template>
  </el-card>

  <section v-if="hasData" class="insight-card">
    <span>系数说明</span>
    <ul>
      <li v-for="statement in notes" :key="statement">{{ statement }}</li>
    </ul>
  </section>
</template>

<style scoped>
.chart-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.chart-card :deep(.el-card__body) {
  padding: 6px 18px 16px;
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

.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  padding: 12px 4px 4px;
}

.summary-item {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 5px;
  padding: 11px 14px;
  border-radius: 12px;
  background: #f8fafc;
}

.summary-item span {
  color: #94a3b8;
  font-size: 11px;
}

.summary-item strong {
  overflow: hidden;
  color: #0f172a;
  font-size: 15px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-item p {
  margin: 0;
  color: #94a3b8;
  font-size: 10px;
}

.fallback-note {
  margin: 10px 4px 0;
  color: #b45309;
  font-size: 11px;
  line-height: 1.6;
}

.chart {
  width: 100%;
}

.chart-empty {
  display: grid;
  min-height: 300px;
  place-items: center;
  padding: 0 10px;
}

.chart-empty :deep(.el-empty__description) {
  max-width: 460px;
}

.chart-empty :deep(.el-empty__description p) {
  color: #64748b;
  font-size: 12px;
  line-height: 1.8;
}

.insight-card {
  padding: 20px 22px;
  border: 1px solid #dbeafe;
  border-radius: 17px;
  background: linear-gradient(135deg, #eff6ff, #f8faff);
}

.insight-card > span {
  color: #3b82f6;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.2px;
}

.insight-card ul {
  padding: 0;
  margin: 10px 0 0;
  list-style: none;
}

.insight-card li {
  position: relative;
  padding-left: 16px;
  margin-bottom: 7px;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.insight-card li:last-child {
  margin-bottom: 0;
}

.insight-card li::before {
  position: absolute;
  top: 8px;
  left: 0;
  width: 5px;
  height: 5px;
  content: '';
  border-radius: 50%;
  background: #3b82f6;
}

@media (max-width: 700px) {
  .summary-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
