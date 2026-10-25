<script setup>
import { computed, ref, watch } from 'vue'

import CoefficientChart from './CoefficientChart.vue'
import DecisionPathExplanation from './DecisionPathExplanation.vue'
import FeatureImportanceChart from './FeatureImportanceChart.vue'
import GaussianNBExplanation from './GaussianNBExplanation.vue'
import GaussianNBLocalLikelihoodChart from './GaussianNBLocalLikelihoodChart.vue'
import KNNExplanation from './KNNExplanation.vue'
import LocalFeatureContributionChart from './LocalFeatureContributionChart.vue'
import MLPArchitecture from './MLPArchitecture.vue'
import MulticlassCoefficientHeatmap from './MulticlassCoefficientHeatmap.vue'
import PermutationImportanceChart from './PermutationImportanceChart.vue'
import SampleExplanationSelector from './SampleExplanationSelector.vue'
import { classDisplayLookup, classDisplayName } from '@/utils/classDisplay'
import { explanationKind, explanationSamples, normalizeExplanationData } from '@/utils/modelExplanation'

const props = defineProps({
  modelName: { type: String, default: '' },
  taskType: { type: String, default: '' },
  trainingInfo: { type: Object, default: null },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const info = computed(() => props.trainingInfo || {})
const kind = computed(() => explanationKind(props.modelName))

// The recorded explanation payload; invalid or legacy records simply leave
// ``explanation.available`` false and the global charts render unchanged.
const explanation = computed(() => normalizeExplanationData(info.value.explanation_data))
const samples = computed(() => explanationSamples(explanation.value))
const hasSamples = computed(() => samples.value.length > 0)
const hasPermutation = computed(
  () => explanation.value.available && explanation.value.kind === 'permutation_importance',
)

const selectedIndex = ref(0)

// A new experiment (or a newly recorded payload) restarts at the first sample.
watch(
  () => props.trainingInfo,
  () => {
    selectedIndex.value = 0
  },
)

const selectedSample = computed(() => {
  if (!hasSamples.value) return null
  const index = Math.min(Math.max(selectedIndex.value, 0), samples.value.length - 1)
  return samples.value[index]
})

const isLogistic = computed(() => props.modelName === 'logistic_regression')

const localContributionTitle = computed(() =>
  isLogistic.value ? '当前样本预测类别线性得分贡献' : '当前样本决策得分贡献',
)

const localContributionSubtitle = computed(() =>
  isLogistic.value
    ? '预测类别的类别线性得分（logit）分解，按贡献绝对值从大到小排列'
    : '预测类别的决策得分分解，按贡献绝对值从大到小排列',
)

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const predictedClassLabel = computed(() => {
  const sample = selectedSample.value
  if (!sample || sample.predicted_class === null || sample.predicted_class === undefined) {
    return ''
  }
  return `预测类别 ${classDisplayName(sample.predicted_class, displayLookup.value)}`
})
</script>

<template>
  <div class="model-explanation">
    <SampleExplanationSelector
      v-if="hasSamples"
      v-model="selectedIndex"
      :samples="samples"
      :task-type="taskType"
      :score-classes="scoreClasses"
      :score-class-names="scoreClassNames"
    />

    <template v-if="kind === 'linear-regression'">
      <div v-if="hasSamples" class="explanation-grid">
        <CoefficientChart
          :coefficients="info.coefficients"
          :feature-names="info.feature_names"
          :intercept="info.intercept"
          :active="active"
        />
        <LocalFeatureContributionChart
          :sample="selectedSample"
          :feature-names="explanation.featureNames"
          title="当前样本特征贡献"
          subtitle="按贡献绝对值从大到小排列，正负贡献使用不同颜色"
          reconstructed-label="还原预测值"
          :active="active"
        />
      </div>
      <CoefficientChart
        v-else
        :coefficients="info.coefficients"
        :feature-names="info.feature_names"
        :intercept="info.intercept"
        :active="active"
      />
    </template>

    <template v-else-if="kind === 'linear-classifier'">
      <MulticlassCoefficientHeatmap
        :coefficients="info.coefficients"
        :feature-names="info.feature_names"
        :classes="info.classes"
        :score-classes="scoreClasses"
        :score-class-names="scoreClassNames"
        :intercept="info.intercept"
        :model-name="modelName"
        :active="active"
      />
      <LocalFeatureContributionChart
        v-if="hasSamples"
        :sample="selectedSample"
        :feature-names="explanation.featureNames"
        :title="localContributionTitle"
        :subtitle="localContributionSubtitle"
        :reconstructed-label="'还原得分'"
        :predicted-label="predictedClassLabel"
        :active="active"
      />
    </template>

    <template v-else-if="kind === 'mlp'">
      <div v-if="hasPermutation" class="explanation-grid">
        <MLPArchitecture :training-info="info" />
        <PermutationImportanceChart :explanation="explanation.raw" :active="active" />
      </div>
      <MLPArchitecture v-else :training-info="info" />
    </template>

    <template v-else-if="kind === 'knn'">
      <KNNExplanation
        :training-info="info"
        :sample="selectedSample"
        :score-classes="scoreClasses"
        :score-class-names="scoreClassNames"
        :active="active"
      />
    </template>

    <template v-else-if="kind === 'gaussian-nb'">
      <GaussianNBExplanation
        :training-info="info"
        :feature-names="info.feature_names"
        :score-classes="scoreClasses"
        :score-class-names="scoreClassNames"
        :active="active"
      />
      <GaussianNBLocalLikelihoodChart
        v-if="hasSamples"
        :sample="selectedSample"
        :feature-names="explanation.featureNames"
        :score-classes="scoreClasses"
        :score-class-names="scoreClassNames"
        :active="active"
      />
    </template>

    <template v-else-if="kind === 'tree'">
      <div v-if="modelName === 'decision_tree_regressor' && hasSamples" class="explanation-grid">
        <FeatureImportanceChart
          :feature-importances="info.feature_importances"
          :feature-names="info.feature_names"
          :active="active"
        />
        <DecisionPathExplanation :sample="selectedSample" />
      </div>

      <div v-else-if="hasPermutation" class="explanation-grid">
        <FeatureImportanceChart
          :feature-importances="info.feature_importances"
          :feature-names="info.feature_names"
          :active="active"
        />
        <PermutationImportanceChart :explanation="explanation.raw" :active="active" />
      </div>

      <FeatureImportanceChart
        v-else
        :feature-importances="info.feature_importances"
        :feature-names="info.feature_names"
        :active="active"
      />
    </template>

    <el-card v-else class="empty-card" shadow="never">
      <div class="empty-body">
        <el-empty :image-size="56" description="当前模型暂无可展示的解释数据。" />
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.model-explanation {
  display: flex;
  flex-direction: column;
  gap: 18px;
  margin-top: 18px;
}

.model-explanation > * {
  min-width: 0;
}

.explanation-grid {
  display: grid;
  align-items: stretch;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.explanation-grid > * {
  min-width: 0;
  height: 100%;
}

/* ---- shared layout styles for explanation child components ---- */

.model-explanation :deep(.card-header) {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.model-explanation :deep(.card-header h3) {
  margin: 0 0 6px;
  color: #1e293b;
  font-size: 15px;
}

.model-explanation :deep(.card-header p) {
  margin: 0;
  color: #94a3b8;
  font-size: 11px;
}

.model-explanation :deep(.fallback-note) {
  margin: 8px 4px 0;
  color: #b45309;
  font-size: 11px;
  line-height: 1.6;
}

.model-explanation :deep(.chart-empty) {
  display: grid;
  place-items: center;
  padding: 0 10px;
}

.model-explanation :deep(.chart-empty.compact) {
  min-height: 180px;
}

.model-explanation :deep(.chart-empty .el-empty__description) {
  max-width: 460px;
}

.model-explanation :deep(.chart-empty .el-empty__description p) {
  color: #64748b;
  font-size: 12px;
  line-height: 1.8;
}

.empty-card :deep(.el-card__body) {
  padding: 10px 16px;
}

.empty-body {
  display: grid;
  min-height: 120px;
  max-height: 160px;
  place-items: center;
}

.empty-body :deep(.el-empty__description p) {
  color: #64748b;
  font-size: 12px;
}

@media (max-width: 1100px) {
  .explanation-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .explanation-grid > * {
    height: auto;
  }
}

@media (max-width: 720px) {
  .model-explanation {
    gap: 14px;
  }

  .explanation-grid {
    gap: 14px;
  }
}
</style>
