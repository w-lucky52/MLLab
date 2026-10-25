<script setup>
import { computed } from 'vue'
import { getAlgorithmTrainingFlow } from '@/utils/training'

const props = defineProps({
  modelName: { type: String, default: '' },
})

const flow = computed(() => getAlgorithmTrainingFlow(props.modelName))
</script>

<template>
  <el-card class="flow-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>{{ flow.title }}</h3>
          <p>按模型类型展示客观训练机制</p>
        </div>

        <el-tag effect="plain" round>{{ flow.steps.length }} 步</el-tag>
      </div>
    </template>

    <ol class="flow-steps">
      <li v-for="(step, index) in flow.steps" :key="index" class="flow-step">
        <span class="step-index">{{ index + 1 }}</span>
        <span class="step-text">{{ step }}</span>
        <span v-if="index < flow.steps.length - 1" class="step-arrow" aria-hidden="true">→</span>
      </li>
    </ol>
  </el-card>
</template>

<style scoped>
.flow-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.flow-card :deep(.el-card__body) {
  padding: 14px 22px 18px;
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

.flow-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 0;
  margin: 0;
  list-style: none;
}

.flow-step {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.step-index {
  display: grid;
  width: 24px;
  height: 24px;
  flex: none;
  color: #2563eb;
  font-size: 12px;
  font-weight: 700;
  border-radius: 8px;
  background: #eff6ff;
  place-items: center;
}

.step-text {
  padding: 7px 12px;
  color: #334155;
  font-size: 12px;
  line-height: 1.5;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #f8fafc;
}

.step-arrow {
  flex: none;
  color: #94a3b8;
  font-size: 13px;
}
</style>
