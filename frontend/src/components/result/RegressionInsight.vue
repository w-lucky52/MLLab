<script setup>
import { computed } from 'vue'
import { DataAnalysis } from '@element-plus/icons-vue'
import { regressionDiagnostics, regressionInsight } from '@/utils/resultDiagnostics'

const props = defineProps({
  yTrue: { type: Array, default: null },
  yPred: { type: Array, default: null },
})

const insight = computed(() =>
  regressionInsight(regressionDiagnostics(props.yTrue, props.yPred)),
)
</script>

<template>
  <section class="analysis-card">
    <div class="analysis-icon">
      <el-icon><DataAnalysis /></el-icon>
    </div>

    <div class="analysis-content">
      <span>结果解读</span>

      <ul v-if="insight.statements.length > 0">
        <li v-for="statement in insight.statements" :key="statement">{{ statement }}</li>
      </ul>

      <p v-else>当前实验数据不足，暂无法生成可靠的解读结论。</p>
    </div>
  </section>
</template>

<style scoped>
.analysis-card {
  display: flex;
  align-items: flex-start;
  gap: 17px;
  padding: 22px 24px;
  border: 1px solid #dbeafe;
  border-radius: 16px;
  background: #f8fbff;
}

.analysis-icon {
  display: grid;
  flex: none;
  width: 42px;
  height: 42px;
  color: #2563eb;
  font-size: 20px;
  border-radius: 12px;
  background: #e0ecff;
  place-items: center;
}

.analysis-content {
  min-width: 0;
}

.analysis-content > span {
  display: block;
  margin-bottom: 8px;
  color: #1e293b;
  font-size: 14px;
  font-weight: 700;
}

.analysis-content ul {
  margin: 0;
  padding-left: 18px;
  color: #475569;
  font-size: 13px;
  line-height: 1.8;
}

.analysis-content p {
  margin: 0;
  color: #94a3b8;
  font-size: 13px;
}
</style>
