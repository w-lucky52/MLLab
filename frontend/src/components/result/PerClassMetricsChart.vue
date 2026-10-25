<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { classDisplayLookup, classDisplayName } from '@/utils/classDisplay'

const props = defineProps({
  perClassMetrics: { type: Array, default: null },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const items = computed(() => (Array.isArray(props.perClassMetrics) ? props.perClassMetrics : []))
const hasData = computed(() => items.value.length > 0)

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const toNumber = (value) => {
  const result = Number(value)
  return Number.isFinite(result) ? result : 0
}

const toPercent = (value) => Number((toNumber(value) * 100).toFixed(2))

const classLabel = (item, index) => {
  if (item?.class !== undefined && item?.class !== null) {
    return classDisplayName(item.class, displayLookup.value)
  }
  return `类别 ${index + 1}`
}

const buildOption = () => {
  const rows = items.value
  const labels = rows.map(classLabel)
  const useZoom = rows.length > 6

  return {
    animationDuration: 700,
    color: ['#2563eb', '#7c3aed', '#059669'],
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
        const row = rows[index]
        if (!row) return ''
        return [
          classLabel(row, index),
          `精确率：${toPercent(row.precision).toFixed(2)}%`,
          `召回率：${toPercent(row.recall).toFixed(2)}%`,
          `F1：${toPercent(row.f1).toFixed(2)}%`,
          `样本数：${toNumber(row.support)}`,
        ].join('<br/>')
      },
    },
    grid: {
      top: 42,
      right: 18,
      bottom: useZoom ? 72 : 30,
      left: 54,
    },
    xAxis: {
      type: 'category',
      data: labels,
      axisTick: { show: false },
      axisLine: { lineStyle: { color: '#dbe3ef' } },
      axisLabel: {
        color: '#64748b',
        fontSize: 11,
        interval: 0,
        rotate: useZoom ? 30 : 0,
      },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100,
      axisLabel: {
        color: '#94a3b8',
        formatter: '{value}%',
      },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    dataZoom: useZoom
      ? [
          { type: 'inside', xAxisIndex: 0 },
          { type: 'slider', xAxisIndex: 0, height: 16, bottom: 10 },
        ]
      : [],
    series: [
      {
        name: '精确率',
        type: 'bar',
        barMaxWidth: 26,
        data: rows.map((row) => toPercent(row.precision)),
      },
      {
        name: '召回率',
        type: 'bar',
        barMaxWidth: 26,
        data: rows.map((row) => toPercent(row.recall)),
      },
      {
        name: 'F1',
        type: 'bar',
        barMaxWidth: 26,
        data: rows.map((row) => toPercent(row.f1)),
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
  id: 'per-class-metrics',
  section: '分类表现',
  title: '每类别分类指标',
  order: 1,
  available: () => hasData.value,
  buildOption,
})

watch(
  () => [props.perClassMetrics, props.scoreClasses, props.scoreClassNames],
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
          <h3>每类别分类指标</h3>
          <p>各个类别的精确率 / 召回率 / F1 对比</p>
        </div>

        <el-tag effect="plain" round>分类结果</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty">
      <el-empty
        :image-size="64"
        description="该历史实验未保存每类别指标，请重新运行实验后查看。"
      />
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
  height: 340px;
}

.chart-empty {
  display: grid;
  min-height: 340px;
  place-items: center;
}
</style>
