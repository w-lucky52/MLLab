<script setup>
import { computed, ref, watch } from 'vue'
import { calculateOvRRoc } from '@/utils/roc'
import { classDisplayLookup, classDisplayName } from '@/utils/classDisplay'
import { useChart } from './useChart'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yScore: { type: Array, default: null },
  scoreType: { type: String, default: null },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const validationIssue = computed(() => {
  const trueLabels = Array.isArray(props.yTrue) ? props.yTrue : []
  const scores = Array.isArray(props.yScore) ? props.yScore : []
  const classes = Array.isArray(props.scoreClasses) ? props.scoreClasses : []

  if (scores.length === 0) {
    return '该历史实验未保存预测分数，请重新运行实验后查看ROC曲线。'
  }
  if (trueLabels.length === 0) {
    return '该历史实验未保存真实标签，请重新运行实验后查看ROC曲线。'
  }
  if (classes.length === 0) {
    return '该实验缺少预测分数对应的类别信息，无法绘制 ROC 曲线。'
  }
  if (scores.length !== trueLabels.length) {
    return '预测分数行数与真实标签数量不一致，无法绘制 ROC 曲线。'
  }
  if (scores.some((row) => !Array.isArray(row) || row.length !== classes.length)) {
    return '预测分数列数与类别数量不一致，无法绘制 ROC 曲线。'
  }
  return ''
})

const calculation = computed(() => {
  if (validationIssue.value) {
    return { issue: validationIssue.value, curves: [] }
  }
  try {
    return {
      issue: '',
      curves: calculateOvRRoc(props.yTrue, props.yScore, props.scoreClasses),
    }
  } catch (error) {
    return { issue: error?.message || '无法绘制 ROC 曲线。', curves: [] }
  }
})

const hasCurves = computed(() => calculation.value.curves.length > 0)

const emptyMessage = computed(() => {
  if (calculation.value.issue) return calculation.value.issue
  if (!hasCurves.value) return '测试集中每个类别都只包含单一标签，无法绘制 ROC 曲线。'
  return ''
})

const scoreTag = computed(() => {
  return props.scoreType === 'decision' ? '决策分数（非概率）' : '预测概率'
})

// Display names only: the ROC calculation keeps using raw scoreClasses.
const curveName = (curve) =>
  `${classDisplayName(curve.classLabel, displayLookup.value)}（AUC ${curve.auc.toFixed(3)}）`

const buildOption = () => {
  const curves = calculation.value.curves

  const series = curves.map((curve) => ({
    name: curveName(curve),
    type: 'line',
    showSymbol: false,
    smooth: false,
    data: curve.fpr.map((fpr, index) => [fpr, curve.tpr[index]]),
    lineStyle: { width: 2 },
  }))

  series.push({
    name: '随机参考线',
    type: 'line',
    data: [
      [0, 0],
      [1, 1],
    ],
    showSymbol: false,
    silent: true,
    tooltip: { show: false },
    lineStyle: { color: '#94a3b8', type: 'dashed', width: 1 },
  })

  return {
    animationDuration: 700,
    color: ['#2563eb', '#7c3aed', '#059669', '#ea580c', '#0891b2', '#db2777', '#65a30d', '#d97706'],
    legend: {
      top: 0,
      type: 'scroll',
      itemWidth: 10,
      itemHeight: 10,
      data: curves.map(curveName),
      textStyle: { color: '#475569', fontSize: 12 },
    },
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        if (!params?.value) return ''
        const fpr = Number(params.value[0]) * 100
        const tpr = Number(params.value[1]) * 100
        return `${params.seriesName}<br/>FPR：${fpr.toFixed(2)}%<br/>TPR：${tpr.toFixed(2)}%`
      },
    },
    grid: { top: 48, right: 24, bottom: 46, left: 58 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: '假阳性率 FPR',
      nameLocation: 'middle',
      nameGap: 30,
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: '真阳性率 TPR',
      nameLocation: 'middle',
      nameGap: 42,
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#edf2f7' } },
    },
    series,
  }
}

const { mount } = useChart(
  chartEl,
  buildOption,
  () => props.active && hasCurves.value,
)

watch(
  () => [props.yTrue, props.yScore, props.scoreClasses, props.scoreClassNames],
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
          <h3>ROC 曲线</h3>
          <p>One-vs-Rest 策略下各类别的假阳性率与真阳性率</p>
        </div>

        <el-tag v-if="hasCurves" effect="plain" round>{{ scoreTag }}</el-tag>
      </div>
    </template>

    <div v-if="!hasCurves" class="chart-empty">
      <el-empty :image-size="64" :description="emptyMessage" />
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
  height: 380px;
}

.chart-empty {
  display: grid;
  min-height: 380px;
  place-items: center;
}
</style>
