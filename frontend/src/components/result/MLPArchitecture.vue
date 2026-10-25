<script setup>
import { computed } from 'vue'
import { Connection } from '@element-plus/icons-vue'
import { summarizeMLPArchitecture } from '@/utils/modelExplanation'

const props = defineProps({
  trainingInfo: { type: Object, default: null },
})

const info = computed(() => props.trainingInfo || {})
const summary = computed(() => summarizeMLPArchitecture(info.value))

const layers = computed(() => {
  const data = summary.value
  return [
    {
      key: 'input',
      label: '输入层',
      value: data.inputText,
      caption: '特征维度',
      tag: '',
    },
    {
      key: 'hidden',
      label: '隐藏层',
      value: data.hiddenText,
      caption: '隐藏单元数量',
      tag: '激活：ReLU',
    },
    {
      key: 'output',
      label: '输出层',
      value: data.outputText,
      caption: '类别数量',
      tag: '输出：Softmax',
    },
  ]
})
</script>

<template>
  <el-card class="structure-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>网络结构</h3>
          <p>单隐藏层全连接网络，维度取自训练时保存的模型参数</p>
        </div>
        <div class="header-tags">
          <el-tag effect="plain" round>{{ summary.structureText }}</el-tag>
          <el-tag effect="plain" round>多层感知机</el-tag>
        </div>
      </div>
    </template>

    <div class="network-flow">
      <div v-for="(layer, index) in layers" :key="layer.key" class="layer-block">
        <div class="layer-card">
          <span class="layer-label">{{ layer.label }}</span>
          <strong class="layer-value">{{ layer.value }}</strong>
          <small>{{ layer.caption }}</small>
          <el-tag v-if="layer.tag" size="small" effect="plain" round>{{ layer.tag }}</el-tag>
        </div>

        <div v-if="index < layers.length - 1" class="layer-arrow">
          <el-icon><Connection /></el-icon>
        </div>
      </div>
    </div>
  </el-card>
</template>

<style scoped>
.structure-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.structure-card :deep(.el-card__body) {
  padding: 20px 22px 18px;
}

.header-tags {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.network-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
}

.layer-block {
  display: flex;
  min-width: 0;
  flex: 1;
  align-items: center;
  gap: 12px;
}

.layer-card {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 20px 14px;
  border: 1px solid #dbeafe;
  border-radius: 15px;
  background: linear-gradient(160deg, #f8fbff, #eff6ff);
  text-align: center;
}

.layer-label {
  color: #64748b;
  font-size: 11px;
}

.layer-value {
  color: #0f172a;
  font-size: 26px;
  font-weight: 750;
  line-height: 1.1;
}

.layer-card small {
  color: #94a3b8;
  font-size: 10px;
}

.layer-arrow {
  flex: none;
  color: #93c5fd;
  font-size: 20px;
}

@media (max-width: 720px) {
  .structure-card :deep(.el-card__body) {
    padding-right: 12px;
    padding-left: 12px;
  }
}

@media (max-width: 700px) {
  .network-flow {
    flex-direction: column;
    align-items: stretch;
  }

  .layer-block {
    flex-direction: column;
  }

  .layer-arrow {
    transform: rotate(90deg);
  }
}
</style>
