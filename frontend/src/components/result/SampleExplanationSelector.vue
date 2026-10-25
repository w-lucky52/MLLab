<script setup>
import { computed } from 'vue'
import { classDisplayLookup, classDisplayName } from '@/utils/classDisplay'
import { formatNumber } from '@/utils/training'

const props = defineProps({
  samples: { type: Array, default: () => [] },
  taskType: { type: String, default: '' },
  scoreClasses: { type: Array, default: null },
  scoreClassNames: { type: Array, default: null },
  modelValue: { type: Number, default: 0 },
})

const emit = defineEmits(['update:modelValue'])

const isClassification = computed(() => props.taskType === 'classification')

const options = computed(() =>
  props.samples.map((sample, index) => ({
    value: index,
    label: `测试样本 #${sample.sample_index}`,
  })),
)

const current = computed(() => props.samples[props.modelValue] ?? null)

const displayLookup = computed(() =>
  classDisplayLookup(props.scoreClasses, props.scoreClassNames),
)

const trueLabel = computed(() => {
  const sample = current.value
  if (!sample) return ''
  if (isClassification.value) return classDisplayName(sample.y_true, displayLookup.value)
  return formatNumber(sample.y_true)
})

const predictedLabel = computed(() => {
  const sample = current.value
  if (!sample) return ''
  if (isClassification.value) return classDisplayName(sample.y_pred, displayLookup.value)
  return formatNumber(sample.y_pred)
})

const onChange = (value) => {
  emit('update:modelValue', value)
}
</script>

<template>
  <el-card v-if="options.length > 0" class="selector-card" shadow="never">
    <div class="selector-row">
      <span class="selector-title">解释对象</span>

      <el-select
        :model-value="modelValue"
        class="selector-dropdown"
        @update:model-value="onChange"
      >
        <el-option
          v-for="option in options"
          :key="option.value"
          :label="option.label"
          :value="option.value"
        />
      </el-select>

      <div v-if="current" class="selector-tags">
        <el-tag effect="plain" round>
          {{ isClassification ? '真实类别' : '真实值' }}：{{ trueLabel }}
        </el-tag>
        <el-tag effect="plain" round type="success">
          {{ isClassification ? '预测类别' : '预测值' }}：{{ predictedLabel }}
        </el-tag>
      </div>
    </div>
  </el-card>
</template>

<style scoped>
.selector-card :deep(.el-card__body) {
  padding: 12px 22px;
}

.selector-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px 16px;
}

.selector-title {
  flex: none;
  color: #1e293b;
  font-size: 14px;
  font-weight: 600;
}

.selector-dropdown {
  width: 180px;
  max-width: 100%;
}

.selector-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

@media (max-width: 720px) {
  .selector-card :deep(.el-card__body) {
    padding: 10px 12px;
  }

  .selector-dropdown {
    width: 150px;
  }
}
</style>
