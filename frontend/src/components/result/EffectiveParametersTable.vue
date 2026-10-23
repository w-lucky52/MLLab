<script setup>
import { computed } from 'vue'
import { parameterSource } from '@/utils/training'

const props = defineProps({
  resolvedParameters: { type: Object, default: null },
  hyperparameters: { type: Object, default: null },
})

const source = computed(() => parameterSource(props.resolvedParameters, props.hyperparameters))
</script>

<template>
  <el-card class="params-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>{{ source.title }}</h3>
          <p v-if="source.kind === 'resolved'">来自本次实验保存的最终生效参数</p>
          <p v-else-if="source.kind === 'submitted'">该实验仅保存了用户提交的参数</p>
          <p v-else>本次实验未保存可用的参数记录</p>
        </div>

        <el-tag v-if="source.note" type="success" effect="plain" round>
          {{ source.note }}
        </el-tag>
      </div>
    </template>

    <div v-if="source.rows.length === 0" class="params-empty">
      <el-empty
        :image-size="64"
        description="该历史实验未记录实际生效参数，请重新运行实验后查看。"
      />
    </div>

    <template v-else>
      <p v-if="source.warning" class="params-warning">{{ source.warning }}</p>

      <div class="params-scroll">
        <table class="params-table">
          <thead>
            <tr>
              <th scope="col">参数名</th>
              <th scope="col">参数值</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in source.rows" :key="row.name">
              <td>{{ row.label }}</td>
              <td>
                <code class="param-value">{{ row.value }}</code>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </el-card>
</template>

<style scoped>
.params-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.params-card :deep(.el-card__body) {
  padding: 14px 22px 20px;
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

.params-warning {
  padding: 11px 14px;
  margin: 0 0 14px;
  color: #b45309;
  font-size: 12px;
  line-height: 1.7;
  border: 1px solid #fde68a;
  border-radius: 11px;
  background: #fffbeb;
}

.params-scroll {
  overflow-x: auto;
}

.params-table {
  width: 100%;
  min-width: 420px;
  border-spacing: 0;
  border-collapse: collapse;
  font-size: 12px;
}

.params-table th {
  padding: 10px 14px;
  color: #64748b;
  font-size: 11px;
  font-weight: 600;
  text-align: left;
  background: #f8fafc;
  border-bottom: 1px solid #e8edf5;
}

.params-table th:first-child {
  border-top-left-radius: 10px;
}

.params-table th:last-child {
  border-top-right-radius: 10px;
}

.params-table td {
  padding: 11px 14px;
  color: #334155;
  border-bottom: 1px solid #edf1f6;
  vertical-align: top;
}

.params-table tbody tr:last-child td {
  border-bottom: 0;
}

.param-value {
  color: #1e293b;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}

.params-empty {
  display: grid;
  min-height: 220px;
  place-items: center;
}

@media (max-width: 700px) {
  .card-header {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
