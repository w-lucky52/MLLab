<script setup>
import { computed } from 'vue'
import { curveStatements, formatNumber, summarizeTrainingCurve } from '@/utils/training'

const props = defineProps({
  lossHistory: { type: Array, default: null },
  lossName: { type: String, default: null },
})

const summary = computed(() => summarizeTrainingCurve(props.lossHistory))
const hasData = computed(() => summary.value.available)
const statements = computed(() => curveStatements(summary.value))

const subtitle = computed(() => {
  if (props.lossName === 'training_loss') return '基于逐轮训练集损失（train_score_）'
  if (props.lossName === 'objective') return '基于逐轮目标函数值（loss_history_）'
  return '基于逐轮损失记录'
})

const formatPercent = (value) => (value === null ? '不适用' : `${Number(value.toFixed(2))}%`)

const statRows = computed(() => {
  if (!summary.value.available) return []
  const stats = summary.value
  return [
    { label: '记录点数', value: String(stats.count) },
    { label: '初始损失', value: formatNumber(stats.initial) },
    { label: '最终损失', value: formatNumber(stats.final) },
    { label: '最小损失', value: formatNumber(stats.minimum) },
    { label: '最小损失轮次', value: `第 ${stats.minRound} 轮` },
    { label: '绝对变化量', value: formatNumber(stats.absoluteChange) },
    { label: '相对变化', value: formatPercent(stats.relativeChangePercent) },
    { label: '末轮与最小损失差值', value: formatNumber(stats.finalVsMinimum) },
  ]
})
</script>

<template>
  <el-card class="summary-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>曲线摘要</h3>
          <p>{{ subtitle }}</p>
        </div>

        <el-tag effect="plain" round>客观统计</el-tag>
      </div>
    </template>

    <div v-if="!hasData" class="summary-empty compact">
      <el-empty
        :image-size="56"
        :description="summary.reason || '该实验未保存训练曲线，无法生成摘要。'"
      />
    </div>

    <template v-else>
      <dl class="stat-list">
        <div v-for="row in statRows" :key="row.label" class="stat-row">
          <dt>{{ row.label }}</dt>
          <dd>{{ row.value }}</dd>
        </div>
      </dl>

      <ul class="statement-list">
        <li v-for="statement in statements" :key="statement">{{ statement }}</li>
      </ul>
    </template>
  </el-card>
</template>

<style scoped>
.summary-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.summary-card :deep(.el-card__body) {
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

.stat-list {
  margin: 0;
}

.stat-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 14px;
  padding: 8px 2px;
  border-bottom: 1px solid #edf1f6;
}

.stat-row:last-child {
  border-bottom: 0;
}

.stat-row dt {
  flex: none;
  color: #94a3b8;
  font-size: 12px;
}

.stat-row dd {
  margin: 0;
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
  overflow-wrap: anywhere;
  text-align: right;
}

.statement-list {
  padding: 10px 0 0;
  margin: 0;
  list-style: none;
}

.statement-list li {
  position: relative;
  padding-left: 16px;
  margin-bottom: 6px;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.statement-list li:last-child {
  margin-bottom: 0;
}

.statement-list li::before {
  position: absolute;
  top: 8px;
  left: 0;
  width: 5px;
  height: 5px;
  content: '';
  border-radius: 50%;
  background: #3b82f6;
}

.summary-empty {
  display: grid;
  place-items: center;
}

.summary-empty.compact {
  min-height: 132px;
}
</style>
