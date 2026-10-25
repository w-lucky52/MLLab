<script setup>
import { computed } from 'vue'
import { ElMessage } from 'element-plus'
import { CopyDocument } from '@element-plus/icons-vue'
import { processLogLines, processLogText } from '@/utils/resultLabels'

const props = defineProps({
  processLog: { type: Array, default: null },
})

// Invalid or missing logs (older experiments) hide the whole card.
const lines = computed(() => processLogLines(props.processLog))
const hasLines = computed(() => lines.value.length > 0)

const copyLog = async () => {
  try {
    await navigator.clipboard.writeText(processLogText(lines.value))
    ElMessage.success('日志已复制')
  } catch {
    ElMessage.error('复制失败，请手动选择文本复制')
  }
}
</script>

<template>
  <el-card v-if="hasLines" class="log-card" shadow="never">
    <template #header>
      <div class="card-header">
        <div>
          <h3>训练日志</h3>
          <p>本次实验的真实执行记录，耗时为相对实验开始的累计时间</p>
        </div>

        <el-button size="small" @click="copyLog">
          <el-icon><CopyDocument /></el-icon>
          复制日志
        </el-button>
      </div>
    </template>

    <div class="log-body">
      <div v-for="line in lines" :key="line.sequence" class="log-line">
        <span class="log-sequence">#{{ line.sequence }}</span>
        <span class="log-phase">[{{ line.phase }}]</span>
        <span class="log-message">{{ line.message }}</span>
        <span v-if="line.elapsedText" class="log-elapsed">{{ line.elapsedText }}</span>
      </div>
    </div>
  </el-card>
</template>

<style scoped>
.log-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.log-card :deep(.el-card__body) {
  padding: 12px 16px 14px;
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

.log-body {
  max-height: 360px;
  padding: 10px 14px;
  overflow-y: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.9;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #f8fafc;
}

.log-line {
  display: flex;
  align-items: baseline;
  gap: 8px;
  min-width: 0;
}

.log-sequence {
  flex: none;
  color: #94a3b8;
}

.log-phase {
  flex: none;
  color: #2563eb;
  font-weight: 700;
}

.log-message {
  min-width: 0;
  color: #334155;
  word-break: break-word;
}

.log-elapsed {
  flex: none;
  margin-left: auto;
  color: #94a3b8;
  font-size: 11px;
}
</style>
