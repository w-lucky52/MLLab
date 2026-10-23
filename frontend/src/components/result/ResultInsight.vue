<script setup>
import { computed } from 'vue'
import { TrendCharts } from '@element-plus/icons-vue'
import { classDisplayLookup, classDisplayName } from '@/utils/classDisplay'

const props = defineProps({
  metrics: { type: Object, default: null },
  perClassMetrics: { type: Array, default: null },
  confusionMatrix: { type: Array, default: null },
  classLabels: { type: Array, default: null },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
})

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const toNumber = (value) => {
  if (value === null || value === undefined || value === '') return null
  const result = Number(value)
  return Number.isFinite(result) ? result : null
}

const formatPercent = (value) => `${((toNumber(value) ?? 0) * 100).toFixed(2)}%`

const labelOf = (labels, index) => {
  if (Array.isArray(labels) && labels[index] !== undefined && labels[index] !== null) {
    return classDisplayName(labels[index], displayLookup.value)
  }
  return `类别 ${index + 1}`
}

const statements = computed(() => {
  const output = []
  const metrics = props.metrics || {}

  const summaryParts = []
  if (toNumber(metrics.accuracy) !== null) {
    summaryParts.push(`准确率（Accuracy）为 ${formatPercent(metrics.accuracy)}`)
  }
  if (toNumber(metrics.f1_weighted) !== null) {
    summaryParts.push(`加权 F1 为 ${formatPercent(metrics.f1_weighted)}`)
  }
  if (summaryParts.length > 0) {
    output.push(`本次实验的${summaryParts.join('，')}。`)
  }

  const perClass = Array.isArray(props.perClassMetrics) ? props.perClassMetrics : []
  if (perClass.length > 0) {
    let lowestIndex = 0
    perClass.forEach((item, index) => {
      const recall = toNumber(item?.recall)
      const lowestRecall = toNumber(perClass[lowestIndex]?.recall)
      if (recall !== null && (lowestRecall === null || recall < lowestRecall)) {
        lowestIndex = index
      }
    })
    output.push(
      `各类别中召回率（Recall）最低的是「${labelOf(
        perClass.map((item) => item?.class),
        lowestIndex,
      )}」，召回率为 ${formatPercent(perClass[lowestIndex]?.recall)}。`,
    )
  }

  const matrix = Array.isArray(props.confusionMatrix) ? props.confusionMatrix : []
  if (matrix.length > 0) {
    let largest = null
    matrix.forEach((row, rowIndex) => {
      if (!Array.isArray(row)) return
      row.forEach((value, columnIndex) => {
        if (rowIndex === columnIndex) return
        const count = toNumber(value)
        if (count !== null && count > 0 && (!largest || count > largest.count)) {
          largest = { rowIndex, columnIndex, count }
        }
      })
    })

    if (largest) {
      output.push(
        `混淆矩阵中最大的误判为：真实类别「${labelOf(
          props.classLabels,
          largest.rowIndex,
        )}」被预测为「${labelOf(props.classLabels, largest.columnIndex)}」，共 ${largest.count} 个样本。`,
      )
    } else {
      output.push('测试集中未出现类别混淆。')
    }
  }

  return output
})
</script>

<template>
  <section class="analysis-card">
    <div class="analysis-icon">
      <el-icon><TrendCharts /></el-icon>
    </div>

    <div class="analysis-content">
      <span>结果解读</span>
      <ul v-if="statements.length > 0">
        <li v-for="statement in statements" :key="statement">{{ statement }}</li>
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
  margin-top: 20px;
  border: 1px solid #dbeafe;
  border-radius: 17px;
  background: linear-gradient(135deg, #eff6ff, #f8faff);
}

.analysis-icon {
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

.analysis-content span {
  color: #3b82f6;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.2px;
}

.analysis-content ul {
  padding: 0;
  margin: 10px 0 0;
  list-style: none;
}

.analysis-content li {
  position: relative;
  padding-left: 16px;
  margin-bottom: 7px;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.analysis-content li::before {
  position: absolute;
  top: 8px;
  left: 0;
  width: 5px;
  height: 5px;
  content: '';
  border-radius: 50%;
  background: #3b82f6;
}

.analysis-content p {
  margin: 10px 0 0;
  color: #64748b;
  font-size: 12px;
  line-height: 1.7;
}
</style>
