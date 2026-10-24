<script setup>
import { computed, ref, watch } from 'vue'
import { useChart } from './useChart'
import { normalizeTrainingSplit } from '@/utils/training'

const props = defineProps({
  trainCount: { type: Number, default: null },
  testCount: { type: Number, default: null },
  active: { type: Boolean, default: true },
})

const chartEl = ref(null)

const split = computed(() => normalizeTrainingSplit(props.trainCount, props.testCount))
const hasData = computed(() => split.value.available)

const buildOption = () => ({
  animationDuration: 700,
  color: ['#2563eb', '#f59e0b'],
  tooltip: {
    trigger: 'item',
    formatter: (params) =>
      `${params.name}：${params.value} 个样本（${params.percent}%）`,
  },
  legend: {
    bottom: 0,
    itemWidth: 10,
    itemHeight: 10,
    textStyle: { color: '#475569', fontSize: 12 },
  },
  series: [
    {
      name: '样本构成',
      type: 'pie',
      radius: ['52%', '74%'],
      center: ['50%', '44%'],
      avoidLabelOverlap: true,
      label: {
        show: true,
        color: '#334155',
        fontSize: 12,
        formatter: '{b}\n{d}%',
      },
      labelLine: { length: 10, length2: 8 },
      data: [
        { name: '训练集', value: split.value.trainCount },
        { name: '测试集', value: split.value.testCount },
      ],
    },
  ],
})

const { mount } = useChart(
  chartEl,
  buildOption,
  () => props.active && hasData.value,
)

watch(
  () => [props.trainCount, props.testCount],
  () => {
    mount()
  },
)
</script>

<template>
  <el-card class="chart-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>训练集 / 测试集构成</h3>
          <p>仅来自本次实验记录的样本数</p>
        </div>

        <el-tag v-if="hasData" effect="plain" round>共 {{ split.total }} 个样本</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="chart-empty compact">
      <el-empty
        :image-size="56"
        :description="split.reason || '该实验未记录训练集与测试集样本数。'"
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
  height: 260px;
}

.chart-empty {
  display: grid;
  place-items: center;
}

.chart-empty.compact {
  min-height: 132px;
}
</style>
