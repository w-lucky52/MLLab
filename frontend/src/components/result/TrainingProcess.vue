<script setup>
import { computed } from 'vue'

import AlgorithmTrainingFlow from './AlgorithmTrainingFlow.vue'
import TrainingLossChart from './TrainingLossChart.vue'
import TrainingProcessLog from './TrainingProcessLog.vue'
import {
  formatDuration,
  formatSampleSplit,
  isIterativeModel,
  modelScaleSummary,
  standardizeSummary,
} from '@/utils/training'

const props = defineProps({
  modelName: { type: String, default: '' },
  hyperparameters: { type: Object, default: null },
  trainingInfo: { type: Object, default: null },
  active: { type: Boolean, default: true },
})

const info = computed(() => props.trainingInfo || {})

const resolvedParameters = computed(() => {
  const value = info.value.resolved_parameters
  return value && typeof value === 'object' && !Array.isArray(value) ? value : null
})

const iterative = computed(() => isIterativeModel(props.modelName))

const summaryCards = computed(() => {
  const scale = modelScaleSummary(
    props.modelName,
    info.value,
    resolvedParameters.value,
    props.hyperparameters,
  )

  return [
    {
      key: 'duration',
      label: '训练耗时',
      value: formatDuration(info.value.fit_duration_ms),
    },
    {
      key: 'split',
      label: '数据划分',
      value: formatSampleSplit(info.value),
    },
    {
      key: 'scale',
      label: scale.label,
      value: scale.value,
    },
    {
      key: 'standardize',
      label: '数据标准化',
      value: standardizeSummary(info.value.standardize),
    },
  ]
})
</script>

<template>
  <div class="training-process">
    <section class="summary-grid">
      <el-card v-for="card in summaryCards" :key="card.key" class="summary-card" shadow="never">
        <span class="summary-label">{{ card.label }}</span>
        <strong class="summary-value">{{ card.value }}</strong>
      </el-card>
    </section>

    <TrainingLossChart
      v-if="iterative"
      :loss-history="info.loss_history"
      :loss-name="info.loss_name"
      :active="active"
    />

    <AlgorithmTrainingFlow :model-name="modelName" />

    <TrainingProcessLog :process-log="info.process_log" />
  </div>
</template>

<style scoped>
.training-process {
  display: flex;
  flex-direction: column;
  gap: 18px;
  margin-top: 18px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.summary-card :deep(.el-card__body) {
  display: flex;
  flex-direction: column;
  gap: 7px;
  min-height: 96px;
  padding: 18px 20px;
}

.summary-label {
  color: #94a3b8;
  font-size: 11px;
}

.summary-value {
  color: #0f172a;
  font-size: 20px;
  font-weight: 750;
  word-break: break-word;
}

.training-process > * {
  min-width: 0;
}

@media (max-width: 1000px) {
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .summary-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
