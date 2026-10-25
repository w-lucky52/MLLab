<script setup>
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { generateZipBlob } from '@/utils/exportZip'

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  meta: { type: Object, default: null },
  items: { type: Array, default: () => [] },
  failedCount: { type: Number, default: 0 },
  rootName: { type: String, default: '' },
})

const emit = defineEmits(['update:modelValue'])

const downloading = ref(false)

const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value),
})

const experimentName = computed(() => props.meta?.name ?? '')
const datasetName = computed(() => props.meta?.datasetName ?? '')
const modelName = computed(() => props.meta?.modelName ?? '')
const chartCount = computed(() => props.items.length)
const zipFileName = computed(() => `${props.rootName}.zip`)

const close = () => {
  visible.value = false
}

const handleDownload = async () => {
  if (downloading.value) return
  downloading.value = true
  try {
    const blob = await generateZipBlob(props.rootName, props.items)
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = zipFileName.value
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
    ElMessage.success('导出成功')
  } catch {
    ElMessage.error('导出失败，请重试')
  } finally {
    downloading.value = false
  }
}
</script>

<template>
  <el-dialog
    v-model="visible"
    class="export-dialog"
    width="860px"
    :show-close="false"
    :close-on-click-modal="false"
    @closed="close"
  >
    <template #header>
      <div class="dialog-header">
        <div>
          <h2>导出结果</h2>
          <p>当前实验的可视化图表</p>
        </div>
        <div class="meta-line">
          <span v-if="experimentName">{{ experimentName }}</span>
          <span v-if="datasetName">{{ datasetName }}</span>
          <span v-if="modelName">{{ modelName }}</span>
        </div>
      </div>
    </template>

    <div class="dialog-body">
      <div class="summary-line">
        <span>已成功生成 {{ chartCount }} 张图表</span>
        <span v-if="failedCount > 0" class="failed-note">有 {{ failedCount }} 张图表未能生成。</span>
      </div>

      <div class="preview-grid">
        <div v-for="item in items" :key="item.filename" class="preview-card">
          <div class="preview-meta">
            <el-tag effect="plain" size="small" round>{{ item.section }}</el-tag>
            <span class="preview-title">{{ item.title }}</span>
          </div>
          <div class="preview-thumb">
            <img :src="item.dataUrl" :alt="item.title" />
          </div>
        </div>
      </div>
    </div>

    <template #footer>
      <div class="dialog-footer">
        <el-button @click="close">关闭</el-button>
        <el-button type="primary" :loading="downloading" @click="handleDownload">
          {{ downloading ? '正在打包…' : '下载 ZIP' }}
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<style scoped>
.dialog-header h2 {
  margin: 0 0 4px;
  color: #0f172a;
  font-size: 18px;
}

.dialog-header p {
  margin: 0;
  color: #94a3b8;
  font-size: 12px;
}

.meta-line {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.meta-line span {
  padding: 3px 10px;
  color: #475569;
  font-size: 12px;
  border-radius: 999px;
  background: #f1f5f9;
}

.dialog-body {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.summary-line {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #334155;
  font-size: 13px;
}

.failed-note {
  color: #b45309;
}

.preview-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  max-height: 65vh;
  overflow-y: auto;
  padding: 2px;
}

.preview-card {
  display: flex;
  min-width: 0;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid #e8edf5;
  border-radius: 12px;
  background: #fff;
}

.preview-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px 8px;
}

.preview-title {
  overflow: hidden;
  color: #1e293b;
  font-size: 12px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.preview-thumb {
  display: grid;
  height: 120px;
  padding: 0 10px 10px;
  place-items: center;
}

.preview-thumb img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

@media (max-width: 1100px) {
  .preview-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .preview-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
