<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { curveSummary, formatNumber, normalizeCurve } from '@/utils/training'

const props = defineProps({
  lossHistory: { type: Array, default: null },
  lossName: { type: String, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const curve = computed(() => normalizeCurve(props.lossHistory))
const hasData = computed(() => Boolean(curve.value))
const stats = computed(() => curveSummary(curve.value))

const axisName = computed(() => {
  if (props.lossName === 'objective') return '目标函数值'
  if (props.lossName === 'training_loss') return '训练损失'
  return '损失值'
})

const subtitle = computed(() => {
  if (props.lossName === 'training_loss') {
    return '逐轮训练集损失（train_score_），不代表测试集误差'
  }
  if (props.lossName === 'objective') {
    return '逐轮目标函数值（loss_history_）'
  }
  return '逐轮损失记录'
})

const statItems = computed(() => {
  const summary = stats.value
  if (!summary) return []

  return [
    { label: '初始值', value: formatNumber(summary.initial) },
    { label: '最终值', value: formatNumber(summary.final) },
    { label: '最小值', value: formatNumber(summary.minimum) },
    { label: '记录点数', value: String(summary.count) },
  ]
})

const buildOption = () => {
  const values = curve.value || []
  const showSymbol = values.length <= 60

  return {
    animationDuration: 700,
    color: ['#2563eb'],
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const first = Array.isArray(params) ? params[0] : params
        const index = first?.dataIndex ?? 0
        const value = values[index]
        if (value === undefined) return ''
        return `轮次：${index + 1}<br/>${axisName.value}：${formatNumber(value)}`
      },
    },
    grid: { top: 28, right: 26, bottom: 78, left: 68 },
    xAxis: {
      type: 'category',
      name: '轮次',
      nameLocation: 'middle',
      nameGap: 46,
      data: values.map((_, index) => String(index + 1)),
      axisTick: { show: false },
      axisLine: { lineStyle: { color: '#dbe3ef' } },
      axisLabel: { color: '#64748b', fontSize: 11, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      name: axisName.value,
      nameLocation: 'middle',
      nameGap: 52,
      scale: true,
      axisLabel: { color: '#94a3b8', fontSize: 11 },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0 },
      { type: 'slider', xAxisIndex: 0, height: 16, bottom: 12 },
    ],
    series: [
      {
        name: axisName.value,
        type: 'line',
        smooth: false,
        showSymbol,
        symbol: 'circle',
        symbolSize: 6,
        data: values,
        lineStyle: { width: 2 },
        itemStyle: { color: '#2563eb' },
      },
    ],
  }
}

const { mount } = useChart(
  chartEl,
  buildOption,
  () => props.active && hasData.value,
)

watch(() => props.lossHistory, () => {
  mount()
}, { deep: true })
</script>

<template>
  <el-card class="chart-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>{{ axisName }}曲线</h3>
          <p>{{ subtitle }}</p>
        </div>

        <el-tag effect="plain" round>{{ lossName === 'training_loss' ? '训练集损失' : '目标函数' }}</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty">
      <el-empty :image-size="64" description="该实验未记录可用的训练曲线。" />
    </div>

    <template v-else>
      <div class="loss-stats">
        <div v-for="item in statItems" :key="item.label" class="loss-stat">
          <span>{{ item.label }}</span>
          <strong>{{ item.value }}</strong>
        </div>
      </div>

      <div ref="chartEl" class="chart"></div>
    </template>
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

.loss-stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  padding: 12px 6px 0;
}

.loss-stat {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 14px;
  border-radius: 12px;
  background: #f8fafc;
}

.loss-stat span {
  color: #94a3b8;
  font-size: 11px;
}

.loss-stat strong {
  color: #0f172a;
  font-size: 15px;
  font-weight: 700;
  word-break: break-all;
}

.chart {
  width: 100%;
  height: 360px;
}

.chart-empty {
  display: grid;
  min-height: 340px;
  place-items: center;
}

@media (max-width: 700px) {
  .loss-stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
