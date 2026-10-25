<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { useExportableChart } from './useExportableChart'
import { formatNumber } from '@/utils/training'
import { classDisplayLookup, classDisplayName } from '@/utils/classDisplay'
import {
  knnExplanationData,
  knnNeighborChartData,
  knnVoteChartData,
} from '@/utils/modelExplanation'

const props = defineProps({
  trainingInfo: { type: Object, default: null },
  // The selected test sample (kind == knn_neighbors); ``null`` for older
  // experiments without recorded neighbours — the compact mechanism card is
  // shown instead.
  sample: { type: Object, default: null },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const neighborChartEl = ref(null)
const voteChartEl = ref(null)

const neighbors = computed(() => knnNeighborChartData(props.sample))
const votes = computed(() => knnVoteChartData(props.sample))
const hasLocal = computed(() => neighbors.value.available && votes.value.available)

const workflow = computed(() => knnExplanationData(props.trainingInfo))

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const predictedLabel = computed(() => {
  if (!votes.value.available) return ''
  return classDisplayName(votes.value.predicted, displayLookup.value)
})

const buildNeighborOption = () => {
  const rows = neighbors.value.rows || []
  const lookup = displayLookup.value

  return {
    animationDuration: 500,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const row = rows[params.dataIndex]
        if (!row) return ''
        return [
          `${row.name}`,
          `训练集内序号：${row.trainIndex}`,
          `类别：${classDisplayName(row.classValue, lookup)}`,
          `距离：${formatNumber(row.distance)}`,
          `投票权重：${formatNumber(row.weight)}`,
        ].join('<br/>')
      },
    },
    grid: { top: 16, right: 34, bottom: 16, left: 12, containLabel: true },
    xAxis: {
      type: 'value',
      name: '距离',
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
      data: rows.map((row) => row.name),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: '#475569', fontSize: 11 },
    },
    series: [
      {
        type: 'bar',
        barMaxWidth: 18,
        data: rows.map((row) => row.distance),
        itemStyle: { borderRadius: [0, 4, 4, 0], color: '#2563eb' },
      },
    ],
  }
}

const buildVoteOption = () => {
  const rows = votes.value.rows || []
  const lookup = displayLookup.value

  return {
    animationDuration: 500,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const row = rows[params.dataIndex]
        if (!row) return ''
        return `类别：${classDisplayName(row.classValue, lookup)}<br/>投票权重合计：${formatNumber(row.weight)}`
      },
    },
    grid: { top: 16, right: 34, bottom: 16, left: 12, containLabel: true },
    xAxis: {
      type: 'value',
      name: '投票权重',
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
      data: rows.map((row) => classDisplayName(row.classValue, lookup)),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: '#475569', fontSize: 11 },
    },
    series: [
      {
        type: 'bar',
        barMaxWidth: 18,
        data: rows.map((row) => row.weight),
        itemStyle: { borderRadius: [0, 4, 4, 0], color: '#059669' },
      },
    ],
  }
}

const isChartActive = () => props.active && hasLocal.value
const neighborChart = useChart(neighborChartEl, buildNeighborOption, isChartActive)
const voteChart = useChart(voteChartEl, buildVoteOption, isChartActive)

useExportableChart({
  id: 'knn-neighbor-distance',
  section: '模型洞察',
  title: '邻居距离',
  order: 7,
  available: () => hasLocal.value,
  buildOption: buildNeighborOption,
})

useExportableChart({
  id: 'knn-votes',
  section: '模型洞察',
  title: '类别投票',
  order: 8,
  available: () => hasLocal.value,
  buildOption: buildVoteOption,
})

watch(
  () => [props.sample, props.scoreClasses, props.scoreClassNames],
  () => {
    neighborChart.mount()
    voteChart.mount()
  },
  { deep: true },
)
</script>

<template>
  <div v-if="hasLocal" class="knn-pair">
    <el-card class="chart-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>邻居距离</h3>
            <p>当前样本的 K 个真实最近邻，按距离从小到大排列</p>
          </div>
          <el-tag effect="plain" round>KNN</el-tag>
        </div>
      </template>

      <div ref="neighborChartEl" class="chart"></div>
    </el-card>

    <el-card class="chart-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>类别投票</h3>
            <p>各类别获得的邻居投票权重合计</p>
          </div>
          <el-tag v-if="predictedLabel" effect="plain" round type="success">
            预测类别 {{ predictedLabel }}
          </el-tag>
        </div>
      </template>

      <div ref="voteChartEl" class="chart"></div>
    </el-card>
  </div>

  <el-card v-else class="workflow-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>KNN 预测机制</h3>
          <p>KNN 在预测时直接使用已保存的训练样本</p>
        </div>
        <el-tag effect="plain" round>K 近邻</el-tag>
      </div>
    </template>

    <ol class="workflow-list">
      <li v-for="(step, index) in workflow.workflow" :key="step">
        <span class="step-index">{{ index + 1 }}</span>
        <p>{{ step }}</p>
      </li>
    </ol>
  </el-card>
</template>

<style scoped>
.knn-pair {
  display: grid;
  align-items: stretch;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.knn-pair > * {
  min-width: 0;
  height: 100%;
}

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

.workflow-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.workflow-card :deep(.el-card__body) {
  padding: 12px 22px 18px;
}

.workflow-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 24px;
  padding: 0;
  margin: 0;
  list-style: none;
}

.workflow-list li {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  min-width: 0;
}

.workflow-list li p {
  margin: 0;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.step-index {
  display: grid;
  width: 22px;
  height: 22px;
  flex: none;
  color: #047857;
  font-size: 11px;
  font-weight: 700;
  border-radius: 50%;
  background: #ecfdf5;
  place-items: center;
}

@media (max-width: 1100px) {
  .knn-pair {
    grid-template-columns: minmax(0, 1fr);
  }

  .knn-pair > * {
    height: auto;
  }
}

@media (max-width: 720px) {
  .chart-card :deep(.el-card__header),
  .workflow-card :deep(.el-card__header) {
    padding: 16px 14px;
  }

  .chart-card :deep(.el-card__body),
  .workflow-card :deep(.el-card__body) {
    padding-right: 12px;
    padding-left: 12px;
  }
}
</style>
