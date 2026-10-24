<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { classDistribution } from '@/utils/resultDiagnostics'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yPred: { type: Array, default: null },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const distribution = computed(() =>
  classDistribution(props.yTrue, props.yPred, props.scoreClasses, props.scoreClassNames),
)

const hasData = computed(
  () => distribution.value.available && distribution.value.classes.length > 0,
)

const buildOption = () => {
  const classes = distribution.value.classes
  const labels = classes.map((item) => item.name)
  const useZoom = classes.length > 8

  return {
    animationDuration: 700,
    color: ['#2563eb', '#f59e0b'],
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
        const item = classes[index]
        if (!item) return ''
        return [
          `类别：${item.name}`,
          `真实数量：${item.actualCount}`,
          `预测数量：${item.predictedCount}`,
        ].join('<br/>')
      },
    },
    grid: {
      top: 36,
      right: 24,
      bottom: useZoom ? 72 : 52,
      left: 46,
    },
    dataZoom: useZoom
      ? [
          { type: 'inside', startValue: 0, endValue: 9 },
          {
            type: 'slider',
            height: 14,
            bottom: 10,
            startValue: 0,
            endValue: 9,
            borderColor: '#e2e8f0',
            fillerColor: 'rgba(37, 99, 235, 0.12)',
            handleSize: '80%',
            textStyle: { color: '#94a3b8', fontSize: 10 },
          },
        ]
      : [],
    xAxis: {
      type: 'category',
      data: labels,
      axisTick: { alignWithLabel: true },
      axisLabel: {
        color: '#64748b',
        fontSize: 11,
        interval: 0,
        rotate: classes.length > 6 ? 30 : 0,
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
        name: '真实数量',
        type: 'bar',
        barMaxWidth: 22,
        data: classes.map((item) => item.actualCount),
      },
      {
        name: '预测数量',
        type: 'bar',
        barMaxWidth: 22,
        data: classes.map((item) => item.predictedCount),
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
  () => [props.yTrue, props.yPred, props.scoreClasses, props.scoreClassNames],
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
          <h3>真实类别 vs 预测类别</h3>
          <p>测试集中每个类别的真实样本数与预测样本数</p>
        </div>

        <el-tag v-if="hasData" effect="plain" round>{{ distribution.classes.length }} 类</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty compact">
      <el-empty :image-size="56" :description="distribution.reason" />
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
