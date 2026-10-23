<script setup>
import { computed } from 'vue'
import { InfoFilled } from '@element-plus/icons-vue'

import CoefficientChart from './CoefficientChart.vue'
import FeatureImportanceChart from './FeatureImportanceChart.vue'
import GaussianNBExplanation from './GaussianNBExplanation.vue'
import KNNExplanation from './KNNExplanation.vue'
import MLPArchitecture from './MLPArchitecture.vue'
import MulticlassCoefficientHeatmap from './MulticlassCoefficientHeatmap.vue'
import { explanationKind } from '@/utils/modelExplanation'

const props = defineProps({
  modelName: { type: String, default: '' },
  taskType: { type: String, default: '' },
  trainingInfo: { type: Object, default: null },
  scoreClasses: { type: Array, default: null },
  active: { type: Boolean, default: true },
})

const info = computed(() => props.trainingInfo || {})
const kind = computed(() => explanationKind(props.modelName))
const standardize = computed(() => info.value.standardize === true)
</script>

<template>
  <div class="model-explanation">
    <section class="explanation-notice">
      <div class="notice-icon">
        <el-icon><InfoFilled /></el-icon>
      </div>

      <div class="notice-content">
        <span>解释说明</span>
        <p>以下内容用于说明模型如何利用输入特征形成预测，不代表变量之间存在因果关系。</p>
      </div>
    </section>

    <template v-if="kind === 'linear-regression'">
      <CoefficientChart
        :coefficients="info.coefficients"
        :feature-names="info.feature_names"
        :intercept="info.intercept"
        :standardize="standardize"
        :active="active"
      />
    </template>

    <template v-else-if="kind === 'linear-classifier'">
      <MulticlassCoefficientHeatmap
        :coefficients="info.coefficients"
        :feature-names="info.feature_names"
        :classes="info.classes"
        :score-classes="scoreClasses"
        :intercept="info.intercept"
        :model-name="modelName"
        :standardize="standardize"
        :active="active"
      />
    </template>

    <template v-else-if="kind === 'mlp'">
      <MLPArchitecture :training-info="info" />
    </template>

    <template v-else-if="kind === 'knn'">
      <KNNExplanation :training-info="info" />
    </template>

    <template v-else-if="kind === 'gaussian-nb'">
      <GaussianNBExplanation
        :training-info="info"
        :feature-names="info.feature_names"
        :score-classes="scoreClasses"
        :active="active"
      />
    </template>

    <template v-else-if="kind === 'tree'">
      <FeatureImportanceChart
        :feature-importances="info.feature_importances"
        :feature-names="info.feature_names"
        :active="active"
      />
    </template>

    <el-card v-else class="empty-card" shadow="never">
      <div class="empty-body">
        <el-empty :image-size="64" description="当前模型暂未提供专用解释视图。" />
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

.explanation-notice {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 18px 22px;
  border: 1px solid #d1fae5;
  border-radius: 17px;
  background: linear-gradient(135deg, #ecfdf5, #f7fefb);
}

.notice-icon {
  display: grid;
  width: 42px;
  height: 42px;
  flex: none;
  color: #047857;
  font-size: 19px;
  border-radius: 12px;
  background: white;
  box-shadow: 0 8px 20px rgb(5 150 105 / 10%);
  place-items: center;
}

.notice-content span {
  color: #047857;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.2px;
}

.notice-content p {
  margin: 8px 0 0;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.empty-card :deep(.el-card__body) {
  padding: 10px 16px;
}

.empty-body {
  display: grid;
  min-height: 240px;
  place-items: center;
}

.empty-body :deep(.el-empty__description p) {
  color: #64748b;
  font-size: 12px;
}

@media (max-width: 640px) {
  .explanation-notice {
    flex-direction: column;
  }
}
</style>
