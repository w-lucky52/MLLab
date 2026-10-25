<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { formatNumber } from '@/utils/training'
import { classDisplayLookup, classDisplayName } from '@/utils/classDisplay'
import { gaussianLikelihoodChartData } from '@/utils/modelExplanation'

const props = defineProps({
  sample: { type: Object, default: null },
  featureNames: { type: Array, default: () => [] },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const data = computed(() =>
  gaussianLikelihoodChartData({ sample: props.sample, featureNames: props.featureNames }),
)

const hasData = computed(() => data.value.available)

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const predictedClassLabel = computed(() =>
  data.value.predictedClass === null
    ? ''
    : classDisplayName(data.value.predictedClass, displayLookup.value),
)

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
          `对数似然：${formatNumber(row.logLikelihood)}`,
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
        data: rows.map((row) => row.logLikelihood),
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: (params) => {
            const row = rows[params.dataIndex]
            if (!row) return '#94a3b8'
            if (row.logLikelihood > 0) return '#059669'
            if (row.logLikelihood < 0) return '#dc2626'
            return '#94a3b8'
          },
        },
      },
    ],
  }
}

const { mount } = useChart(chartEl, buildOption, () => props.active && hasData.value)

useExportableChart({
  id: 'gnb-local-likelihood',
  section: '模型洞察',
  title: '当前样本特征对数似然',
  order: 9,
  available: () => hasData.value,
  buildOption,
})

watch(
  () => [props.sample, props.featureNames, props.scoreClasses, props.scoreClassNames],
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
          <h3>当前样本特征对数似然</h3>
          <p>在预测类别下，各特征取值对应的高斯对数似然</p>
        </div>
        <div class="header-tags">
          <el-tag v-if="predictedClassLabel" effect="plain" round type="success">
            预测类别 {{ predictedClassLabel }}
          </el-tag>
          <el-tag
            v-if="data.predictedProbability !== null"
            effect="plain"
            round
          >
            预测类别概率 {{ formatNumber(data.predictedProbability) }}
          </el-tag>
          <el-tag effect="plain" round>log_prior {{ formatNumber(data.logPrior) }}</el-tag>
          <el-tag effect="plain" round>
            joint_log_likelihood {{ formatNumber(data.joint) }}
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
