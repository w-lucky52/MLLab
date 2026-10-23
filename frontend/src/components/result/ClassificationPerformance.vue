<script setup>
import ConfusionMatrixChart from './ConfusionMatrixChart.vue'
import PerClassMetricsChart from './PerClassMetricsChart.vue'
import ResultInsight from './ResultInsight.vue'
import RocCurveChart from './RocCurveChart.vue'

const props = defineProps({
  evaluation: { type: Object, default: null },
  active: { type: Boolean, default: true },
})

const field = (key) => props.evaluation?.[key] ?? null
</script>

<template>
  <div class="classification-performance">
    <PerClassMetricsChart
      :per-class-metrics="field('per_class_metrics')"
      :score-classes="field('score_classes')"
      :score-class-names="field('score_class_names')"
      :active="active"
    />

    <div class="performance-row">
      <ConfusionMatrixChart
        :confusion-matrix="field('confusion_matrix')"
        :class-labels="field('class_labels')"
        :score-classes="field('score_classes')"
        :score-class-names="field('score_class_names')"
        :active="active"
      />

      <RocCurveChart
        :y-true="field('y_true')"
        :y-score="field('y_score')"
        :score-type="field('score_type')"
        :score-classes="field('score_classes')"
        :score-class-names="field('score_class_names')"
        :active="active"
      />
    </div>

    <ResultInsight
      :metrics="field('metrics')"
      :per-class-metrics="field('per_class_metrics')"
      :confusion-matrix="field('confusion_matrix')"
      :class-labels="field('class_labels')"
      :score-classes="field('score_classes')"
      :score-class-names="field('score_class_names')"
    />
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
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

@media (max-width: 1000px) {
  .performance-row {
    grid-template-columns: 1fr;
  }
}
</style>
