<script setup>
import { computed } from 'vue'
import { Connection } from '@element-plus/icons-vue'
import { parameterRows } from '@/utils/training'
import { explanationNotes, mlpDimensions } from '@/utils/modelExplanation'

const props = defineProps({
  trainingInfo: { type: Object, default: null },
})

const info = computed(() => props.trainingInfo || {})
const dimensions = computed(() => mlpDimensions(info.value))
const notes = computed(() => explanationNotes('mlp', info.value))

const parameterRowsList = computed(() => {
  const resolved = info.value.resolved_parameters
  const value = resolved && typeof resolved === 'object' && !Array.isArray(resolved) ? resolved : null
  return parameterRows(value)
})

const layers = computed(() => {
  const data = dimensions.value
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
  <div class="mlp-explanation">
    <el-card class="structure-card" shadow="never">
      <template #header>
        <div class="card-header">
          <div>
            <h3>网络结构</h3>
            <p>单隐藏层全连接网络，结构信息来自本次训练记录</p>
          </div>
          <el-tag effect="plain" round>多层感知机</el-tag>
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

      <p v-if="!dimensions.valid" class="dimension-warning">
        部分网络维度未记录，参数总量无法计算，上方显示为“未记录”。
      </p>
    </el-card>

    <div class="detail-row">
      <el-card class="parameter-card" shadow="never">
        <template #header>
          <div class="card-header">
            <div>
              <h3>参数总量</h3>
              <p>含偏置项，仅统计全连接层的可学习参数</p>
            </div>
          </div>
        </template>

        <template v-if="dimensions.valid">
          <div class="total-value">
            <strong>{{ dimensions.parameterText }}</strong>
            <span>个参数</span>
          </div>

          <div class="breakdown">
            <div v-for="item in dimensions.breakdown" :key="item.label" class="breakdown-item">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </div>
          </div>

          <p class="formula">
            计算方式：输入维度 × 隐藏单元数 + 隐藏单元数 + 隐藏单元数 × 输出维度 + 输出维度
          </p>
        </template>

        <el-empty v-else :image-size="48" description="网络维度不完整，未计算参数总量。" />
      </el-card>

      <el-card class="parameter-card" shadow="never">
        <template #header>
          <div class="card-header">
            <div>
              <h3>训练参数</h3>
              <p>本次训练实际生效的构造参数</p>
            </div>
          </div>
        </template>

        <el-table v-if="parameterRowsList.length" :data="parameterRowsList" size="small">
          <el-table-column prop="label" label="参数" min-width="140" show-overflow-tooltip />
          <el-table-column prop="value" label="实际值" min-width="100" align="right" />
        </el-table>

        <el-empty v-else :image-size="48" description="该实验未记录实际生效参数。" />
      </el-card>
    </div>

    <section class="insight-card">
      <span>结构说明</span>
      <ul>
        <li v-for="statement in notes" :key="statement">{{ statement }}</li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.mlp-explanation {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.structure-card :deep(.el-card__header),
.parameter-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.structure-card :deep(.el-card__body) {
  padding: 20px 22px 18px;
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

.dimension-warning {
  margin: 16px 0 0;
  padding: 10px 14px;
  color: #b45309;
  font-size: 11px;
  line-height: 1.6;
  border-radius: 10px;
  background: #fffbeb;
}

.detail-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 18px;
}

.parameter-card :deep(.el-card__body) {
  padding: 12px 18px 16px;
}

.total-value {
  display: flex;
  align-items: baseline;
  gap: 7px;
  padding: 4px 2px 12px;
}

.total-value strong {
  color: #047857;
  font-size: 34px;
  font-weight: 780;
}

.total-value span {
  color: #64748b;
  font-size: 12px;
}

.breakdown {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.breakdown-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 9px 12px;
  border-radius: 10px;
  background: #f8fafc;
}

.breakdown-item span {
  color: #94a3b8;
  font-size: 11px;
}

.breakdown-item strong {
  color: #334155;
  font-size: 13px;
}

.formula {
  margin: 12px 2px 0;
  color: #94a3b8;
  font-size: 10px;
  line-height: 1.7;
}

.parameter-card :deep(.el-table__cell) {
  font-size: 12px;
}

.insight-card {
  padding: 20px 22px;
  border: 1px solid #dbeafe;
  border-radius: 17px;
  background: linear-gradient(135deg, #eff6ff, #f8faff);
}

.insight-card > span {
  color: #3b82f6;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.2px;
}

.insight-card ul {
  padding: 0;
  margin: 10px 0 0;
  list-style: none;
}

.insight-card li {
  position: relative;
  padding-left: 16px;
  margin-bottom: 7px;
  color: #475569;
  font-size: 12px;
  line-height: 1.7;
}

.insight-card li:last-child {
  margin-bottom: 0;
}

.insight-card li::before {
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
  .detail-row {
    grid-template-columns: minmax(0, 1fr);
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
