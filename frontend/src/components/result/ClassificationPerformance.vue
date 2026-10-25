<script setup>
import ConfusionMatrixChart from './ConfusionMatrixChart.vue'
import PerClassMetricsChart from './PerClassMetricsChart.vue'
import PredictionConfidenceChart from './PredictionConfidenceChart.vue'
import RocCurveChart from './RocCurveChart.vue'

const props = defineProps({
  evaluation: { type: Object, default: null },
  active: { type: Boolean, default: true },
})

const field = (key) => props.evaluation?.[key] ?? null
</script>

<template>
  <div class="classification-performance">
    <div class="performance-row">
      <PerClassMetricsChart
        :per-class-metrics="field('per_class_metrics')"
        :score-classes="field('score_classes')"
        :score-class-names="field('score_class_names')"
        :active="active"
      />

      <ConfusionMatrixChart
        :confusion-matrix="field('confusion_matrix')"
        :class-labels="field('class_labels')"
        :score-classes="field('score_classes')"
        :score-class-names="field('score_class_names')"
        :active="active"
      />
    </div>

    <div class="performance-row">
      <RocCurveChart
        :y-true="field('y_true')"
        :y-score="field('y_score')"
        :score-type="field('score_type')"
        :score-classes="field('score_classes')"
        :score-class-names="field('score_class_names')"
        :active="active"
      />

      <PredictionConfidenceChart
        :y-true="field('y_true')"
        :y-pred="field('y_pred')"
        :y-score="field('y_score')"
        :score-type="field('score_type')"
        :active="active"
      />
    </div>
  </div>
</template>

<style scoped>
.classification-performance {
  display: flex;
  flex-direction: column;
  gap: 18px;
  margin-top: 18px;
}

.performance-row {
  display: grid;
  align-items: stretch;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.performance-row > * {
  min-width: 0;
  height: 100%;
}

.classification-performance :deep(.chart) {
  height: 350px;
}

.classification-performance :deep(.chart-empty:not(.compact)) {
  min-height: 350px;
}

@media (max-width: 1100px) {
  .performance-row {
    grid-template-columns: minmax(0, 1fr);
  }

  .performance-row > * {
    height: auto;
  }
}
</style>
