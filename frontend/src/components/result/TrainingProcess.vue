<script setup>
import { computed } from 'vue'
import { TrendCharts } from '@element-plus/icons-vue'

import AlgorithmTrainingFlow from './AlgorithmTrainingFlow.vue'
import EffectiveParametersTable from './EffectiveParametersTable.vue'
import TrainingCurveSummary from './TrainingCurveSummary.vue'
import TrainingLossChart from './TrainingLossChart.vue'
import TrainingSplitChart from './TrainingSplitChart.vue'
import {
  MISSING_CURVE_HINT,
  formatDuration,
  formatSampleSplit,
  getNoCurveExplanation,
  isIterativeModel,
  iterationSummary,
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
const hasCurve = computed(() => Boolean(curve.value))
const iterative = computed(() => isIterativeModel(props.modelName))
const noCurveStatements = computed(() => getNoCurveExplanation(props.modelName))

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

    <template v-if="hasCurve">
      <div class="training-row">
        <TrainingLossChart
          :loss-history="info.loss_history"
          :loss-name="info.loss_name"
          :active="active"
        />

        <TrainingSplitChart
          :train-count="info.train_sample_count"
          :test-count="info.test_sample_count"
          :active="active"
        />
      </div>

      <div class="training-row">
        <EffectiveParametersTable
          :resolved-parameters="resolvedParameters"
          :hyperparameters="hyperparameters"
        />

        <TrainingCurveSummary :loss-history="info.loss_history" :loss-name="info.loss_name" />
      </div>

      <AlgorithmTrainingFlow :model-name="modelName" :parameters="resolvedParameters" />
    </template>

    <template v-else-if="iterative">
      <div class="training-row">
        <el-card class="hint-card" shadow="never">
          <div class="hint-body">
            <el-empty :image-size="56" :description="MISSING_CURVE_HINT" />
          </div>
        </el-card>

        <TrainingSplitChart
          :train-count="info.train_sample_count"
          :test-count="info.test_sample_count"
          :active="active"
        />
      </div>

      <div class="training-row">
        <EffectiveParametersTable
          :resolved-parameters="resolvedParameters"
          :hyperparameters="hyperparameters"
        />

        <TrainingCurveSummary :loss-history="null" :loss-name="info.loss_name" />
      </div>

      <AlgorithmTrainingFlow :model-name="modelName" :parameters="resolvedParameters" />
    </template>

    <template v-else>
      <div class="training-row">
        <AlgorithmTrainingFlow :model-name="modelName" :parameters="resolvedParameters" />

        <TrainingSplitChart
          :train-count="info.train_sample_count"
          :test-count="info.test_sample_count"
          :active="active"
        />
      </div>

      <div class="training-row">
        <EffectiveParametersTable
          :resolved-parameters="resolvedParameters"
          :hyperparameters="hyperparameters"
        />

        <el-card class="note-card" shadow="never">
          <template #header>
            <div class="card-header">
              <div>
                <h3>为什么没有损失曲线</h3>
                <p>该模型的训练机制与逐轮优化无关</p>
              </div>
            </div>
          </template>

          <ul class="note-list">
            <li v-for="statement in noCurveStatements" :key="statement">{{ statement }}</li>
          </ul>
        </el-card>
      </div>
    </template>

    <section v-if="iterative" class="insight-card">
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

.training-row {
  display: grid;
  align-items: start;
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
  gap: 18px;
}

.training-row > * {
  min-width: 0;
}

.hint-card :deep(.el-card__body) {
  padding: 10px 16px;
}

.hint-body {
  display: grid;
  min-height: 132px;
  place-items: center;
}

.note-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.note-card :deep(.el-card__body) {
  padding: 12px 22px 18px;
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

.note-list {
  padding: 0;
  margin: 0;
  list-style: none;
}

.note-list li {
  position: relative;
  padding-left: 16px;
  margin-bottom: 8px;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.note-list li:last-child {
  margin-bottom: 0;
}

.note-list li::before {
  position: absolute;
  top: 8px;
  left: 0;
  width: 5px;
  height: 5px;
  content: '';
  border-radius: 50%;
  background: #3b82f6;
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

@media (max-width: 1100px) {
  .training-row {
    grid-template-columns: minmax(0, 1fr);
  }
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
