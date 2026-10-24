<script setup>
import { computed } from 'vue'
import ActualPredictedChart from './ActualPredictedChart.vue'
import RegressionInsight from './RegressionInsight.vue'
import ResidualHistogramChart from './ResidualHistogramChart.vue'
import ResidualScatterChart from './ResidualScatterChart.vue'
import TopAbsoluteErrorsChart from './TopAbsoluteErrorsChart.vue'
import { regressionDiagnostics } from '@/utils/resultDiagnostics'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yPred: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const diagnostics = computed(() => regressionDiagnostics(props.yTrue, props.yPred))
const hasData = computed(() => diagnostics.value.available)
</script>

<template>
  <div class="regression-performance">
    <el-card v-if="!hasData" class="chart-card" shadow="never">
      <div class="chart-empty compact">
        <el-empty
          :image-size="56"
          :description="diagnostics.reason || '该实验未保存真实值与预测值，无法绘制诊断图。'"
        />
      </div>
    </el-card>

    <template v-else>
      <div class="performance-row wide-left">
        <ActualPredictedChart :y-true="yTrue" :y-pred="yPred" :active="active" />
        <ResidualHistogramChart :y-true="yTrue" :y-pred="yPred" :active="active" />
      </div>

      <div class="performance-row wide-left">
        <ResidualScatterChart :y-true="yTrue" :y-pred="yPred" :active="active" />
        <TopAbsoluteErrorsChart :y-true="yTrue" :y-pred="yPred" :active="active" />
      </div>

      <RegressionInsight :y-true="yTrue" :y-pred="yPred" />
    </template>
  </div>
</template>

<style scoped>
.regression-performance {
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

.performance-row > * {
  min-width: 0;
}

.performance-row.wide-left {
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
}

.chart-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.chart-card :deep(.el-card__body) {
  padding: 5px 16px 15px;
}

.chart-empty {
  display: grid;
  min-height: 320px;
  place-items: center;
}

.chart-empty.compact {
  min-height: 132px;
}

@media (max-width: 1100px) {
  .performance-row,
  .performance-row.wide-left {
    grid-template-columns: 1fr;
  }
}
</style>
