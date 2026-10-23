<script setup>
import { computed } from 'vue'
import { Position } from '@element-plus/icons-vue'
import { explanationNotes, knnExplanationData } from '@/utils/modelExplanation'

const props = defineProps({
  trainingInfo: { type: Object, default: null },
})

const data = computed(() => knnExplanationData(props.trainingInfo))
const notes = computed(() => explanationNotes('knn', props.trainingInfo))

const cards = computed(() => [
  { key: 'neighbors', label: '邻居数量 K', value: data.value.neighborsText },
  { key: 'weights', label: '权重策略', value: data.value.weightsText },
  { key: 'distance', label: '距离度量', value: data.value.distanceText },
  { key: 'samples', label: '已保存训练样本', value: data.value.trainSamplesText },
])
</script>

<template>
  <div class="knn-explanation">
    <el-card class="mechanism-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>KNN 工作机制</h3>
            <p>机制参数来自本次训练记录，缺失时显示“未记录”</p>
          </div>
          <el-tag effect="plain" round>K 近邻</el-tag>
        </div>
      </template>

      <section class="summary-grid">
        <div v-for="card in cards" :key="card.key" class="summary-item">
          <span>{{ card.label }}</span>
          <strong>{{ card.value }}</strong>
        </div>
      </section>
    </el-card>

    <el-card class="workflow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>预测流程</h3>
            <p>KNN 在预测时直接使用已保存的训练样本</p>
          </div>
        </div>
      </template>

      <ol class="workflow-list">
        <li v-for="(step, index) in data.workflow" :key="step">
          <span class="step-index">{{ index + 1 }}</span>
          <p>{{ step }}</p>
        </li>
      </ol>
    </el-card>

    <section class="insight-card">
      <div class="insight-icon">
        <el-icon><Position /></el-icon>
      </div>

      <div class="insight-content">
        <span>模型说明</span>
        <ul>
          <li v-for="statement in notes" :key="statement">{{ statement }}</li>
        </ul>
      </div>
    </section>
  </div>
</template>

<style scoped>
.knn-explanation {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.mechanism-card :deep(.el-card__header),
.workflow-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.mechanism-card :deep(.el-card__body) {
  padding: 8px 18px 18px;
}

.workflow-card :deep(.el-card__body) {
  padding: 12px 22px 18px;
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

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  padding: 6px 4px 0;
}

.summary-item {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 6px;
  padding: 14px 16px;
  border-radius: 12px;
  background: #f8fafc;
}

.summary-item span {
  color: #94a3b8;
  font-size: 11px;
}

.summary-item strong {
  overflow: hidden;
  color: #0f172a;
  font-size: 15px;
  font-weight: 700;
  line-height: 1.5;
  text-overflow: ellipsis;
}

.workflow-list {
  padding: 0;
  margin: 0;
  list-style: none;
}

.workflow-list li {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 9px 0;
}

.workflow-list li p {
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
  color: #047857;
  font-size: 11px;
  font-weight: 700;
  border-radius: 50%;
  background: #ecfdf5;
  place-items: center;
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
