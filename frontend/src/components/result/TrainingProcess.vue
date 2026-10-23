<script setup>
import { computed } from 'vue'
import { TrendCharts } from '@element-plus/icons-vue'

import EffectiveParametersTable from './EffectiveParametersTable.vue'
import TrainingLossChart from './TrainingLossChart.vue'
import {
  formatDuration,
  formatSampleSplit,
  iterationSummary,
  lossEmptyDescription,
  normalizeCurve,
  standardizeSummary,
  stopReasonText,
  trainingInsights,
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

const curve = computed(() => normalizeCurve(info.value.loss_history))
const emptyDescription = computed(() => lossEmptyDescription(props.modelName))

const summaryCards = computed(() => {
  const standardize = standardizeSummary(info.value.standardize)

  return [
    {
      key: 'duration',
      label: '训练耗时',
      value: formatDuration(info.value.fit_duration_ms),
      hint: '',
    },
    {
      key: 'split',
      label: '数据划分',
      value: formatSampleSplit(info.value),
      hint: '',
    },
    {
      key: 'iteration',
      label: '迭代情况',
      value: iterationSummary(info.value, resolvedParameters.value),
      hint: stopReasonText(info.value.stop_reason),
    },
    {
      key: 'standardize',
      label: '数据标准化',
      value: standardize.text,
      hint: standardize.hint,
    },
  ]
})

const insights = computed(() =>
  trainingInsights({
    trainingInfo: info.value,
    modelName: props.modelName,
    curve: curve.value,
  }),
)
</script>

<template>
  <div class="training-process">
    <section class="summary-grid">
      <el-card v-for="card in summaryCards" :key="card.key" class="summary-card" shadow="never">
        <span class="summary-label">{{ card.label }}</span>
        <strong class="summary-value">{{ card.value }}</strong>
        <p v-if="card.hint" class="summary-hint">{{ card.hint }}</p>
      </el-card>
    </section>

    <TrainingLossChart
      v-if="curve"
      :loss-history="info.loss_history"
      :loss-name="info.loss_name"
      :active="active"
    />

    <el-card v-else class="empty-card" shadow="never">
      <div class="empty-body">
        <el-empty :image-size="64" :description="emptyDescription" />
      </div>
    </el-card>

    <EffectiveParametersTable
      :resolved-parameters="resolvedParameters"
      :hyperparameters="hyperparameters"
    />

    <section class="insight-card">
      <div class="insight-icon">
        <el-icon><TrendCharts /></el-icon>
      </div>

      <div class="insight-content">
        <span>训练过程说明</span>
        <ul>
          <li v-for="statement in insights" :key="statement">{{ statement }}</li>
        </ul>
      </div>
    </section>
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
  padding: 18px 20px;
  min-height: 96px;
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

.summary-hint {
  margin: 0;
  color: #94a3b8;
  font-size: 10px;
  line-height: 1.6;
}

.empty-card :deep(.el-card__body) {
  padding: 10px 16px;
}

.empty-body {
  display: grid;
  min-height: 260px;
  place-items: center;
  padding: 0 8px;
}

.empty-body :deep(.el-empty__description) {
  max-width: 460px;
}

.empty-body :deep(.el-empty__description p) {
  color: #64748b;
  font-size: 12px;
  line-height: 1.8;
}

.insight-card {
  display: flex;
  align-items: flex-start;
  gap: 17px;
  padding: 22px 24px;
  border: 1px solid #dbeafe;
  border-radius: 17px;
  background: linear-gradient(135deg, #eff6ff, #f8faff);
}

.insight-icon {
  display: grid;
  width: 46px;
  height: 46px;
  flex: none;
  color: #2563eb;
  font-size: 21px;
  border-radius: 14px;
  background: white;
  box-shadow: 0 8px 20px rgb(37 99 235 / 10%);
  place-items: center;
}

.insight-content span {
  color: #3b82f6;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.2px;
}

.insight-content ul {
  padding: 0;
  margin: 10px 0 0;
  list-style: none;
}

.insight-content li {
  position: relative;
  padding-left: 16px;
  margin-bottom: 7px;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.insight-content li:last-child {
  margin-bottom: 0;
}

.insight-content li::before {
  position: absolute;
  top: 8px;
  left: 0;
  width: 5px;
  height: 5px;
  content: '';
  border-radius: 50%;
  background: #3b82f6;
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

  .insight-card {
    flex-direction: column;
  }
}
</style>
