<script setup>
import { computed } from 'vue'
import { decisionPathRows } from '@/utils/modelExplanation'
import { formatNumber } from '@/utils/training'

const props = defineProps({
  sample: { type: Object, default: null },
})

const data = computed(() => decisionPathRows(props.sample))
const hasData = computed(() => data.value.available)
</script>

<template>
  <el-card v-if="hasData" class="path-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>当前样本决策路径</h3>
          <p>从根节点到叶节点的真实遍历过程</p>
        </div>
        <el-tag effect="plain" round>{{ data.steps.length }} 步</el-tag>
      </div>
    </template>

    <div class="path-body">
      <ol class="path-list">
        <li v-for="step in data.steps" :key="step.step" class="path-step">
          <span class="step-index">{{ step.step }}</span>
          <p>{{ step.text }}</p>
        </li>
        <li class="path-leaf">
          <span class="leaf-dot"></span>
          <p>叶节点预测值：{{ formatNumber(data.leafPrediction) }}</p>
          <el-tag effect="plain" round type="success">
            预测值 {{ formatNumber(data.leafPrediction) }}
          </el-tag>
        </li>
      </ol>
    </div>
  </el-card>
</template>

<style scoped>
.path-card {
  display: flex;
  flex-direction: column;
}

.path-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.path-card :deep(.el-card__body) {
  min-height: 0;
  padding: 8px 22px 16px;
}

.path-body {
  height: 350px;
  overflow-y: auto;
  padding-right: 6px;
}

.path-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 4px 0 0;
  margin: 0;
  list-style: none;
}

.path-step {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.path-step p {
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
  color: #1d4ed8;
  font-size: 11px;
  font-weight: 700;
  border-radius: 50%;
  background: #eff6ff;
  place-items: center;
}

.path-leaf {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-top: 6px;
  border-top: 1px dashed #e2e8f0;
}

.path-leaf p {
  margin: 0;
  color: #0f172a;
  font-size: 12px;
  font-weight: 600;
}

.leaf-dot {
  width: 10px;
  height: 10px;
  flex: none;
  margin: 0 6px;
  border-radius: 50%;
  background: #059669;
}

@media (max-width: 720px) {
  .path-card :deep(.el-card__header) {
    padding: 16px 14px;
  }

  .path-card :deep(.el-card__body) {
    padding-right: 12px;
    padding-left: 12px;
  }
}
</style>
