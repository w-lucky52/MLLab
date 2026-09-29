<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowRight, Check, DataBoard, Document, UploadFilled } from '@element-plus/icons-vue'

const router = useRouter()

const selectedDataset = ref('')
const fileName = ref('')

const datasets = [
  {
    name: '鸢尾花数据集',
    value: 'iris',
    description: '根据花萼和花瓣特征识别鸢尾花类别。',
    samples: 150,
    features: 4,
    task: '多分类',
    color: 'blue',
  },
  {
    name: '葡萄酒数据集',
    value: 'wine',
    description: '根据化学成分特征识别葡萄酒类别。',
    samples: 178,
    features: 13,
    task: '多分类',
    color: 'purple',
  },
  {
    name: '乳腺癌数据集',
    value: 'breast-cancer',
    description: '根据细胞特征判断肿瘤的良性与恶性。',
    samples: 569,
    features: 30,
    task: '二分类',
    color: 'green',
  },
]

const currentDataset = computed(() => {
  return datasets.find((item) => item.value === selectedDataset.value)
})

const selectDataset = (dataset) => {
  selectedDataset.value = dataset.value
  fileName.value = ''
}

const handleFileChange = (uploadFile) => {
  if (!uploadFile.name.toLowerCase().endsWith('.csv')) {
    ElMessage.error('只能选择 CSV 格式的文件')
    return
  }

  fileName.value = uploadFile.name
  selectedDataset.value = ''
  ElMessage.success(`已选择文件：${uploadFile.name}`)
}

const confirmDataset = () => {
  if (!selectedDataset.value && !fileName.value) {
    ElMessage.warning('请先选择数据集或上传 CSV 文件')
    return
  }

  ElMessage.success('数据集选择成功')

  router.push({
    path: '/experiment',
    query: {
      dataset: selectedDataset.value || fileName.value,
    },
  })
}
</script>

<template>
  <div class="data-page">
    <section class="page-heading">
      <div>
        <span class="eyebrow">DATA CENTER</span>
        <h1>选择实验数据</h1>
        <p>从标准数据集开始，或者上传自己的 CSV 数据文件。</p>
      </div>

      <div class="heading-icon">
        <el-icon><DataBoard /></el-icon>
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <div>
          <h2>内置数据集</h2>
          <p>经过整理的经典机器学习教学数据集</p>
        </div>

        <el-tag effect="plain" round> 共 {{ datasets.length }} 个数据集 </el-tag>
      </div>

      <div class="dataset-grid">
        <article
          v-for="dataset in datasets"
          :key="dataset.value"
          :class="['dataset-card', dataset.color, { selected: selectedDataset === dataset.value }]"
          @click="selectDataset(dataset)"
        >
          <div class="dataset-top">
            <div class="dataset-icon">
              <el-icon><Document /></el-icon>
            </div>

            <div v-if="selectedDataset === dataset.value" class="selected-mark">
              <el-icon><Check /></el-icon>
            </div>
          </div>

          <h3>{{ dataset.name }}</h3>
          <p>{{ dataset.description }}</p>

          <div class="dataset-meta">
            <div>
              <span>样本数量</span>
              <strong>{{ dataset.samples }}</strong>
            </div>

            <div>
              <span>特征数量</span>
              <strong>{{ dataset.features }}</strong>
            </div>

            <div>
              <span>任务类型</span>
              <strong>{{ dataset.task }}</strong>
            </div>
          </div>
        </article>
      </div>
    </section>

    <div class="divider">
      <span>或者使用自己的数据</span>
    </div>

    <section class="upload-section">
      <el-upload
        drag
        accept=".csv"
        :auto-upload="false"
        :limit="1"
        :show-file-list="false"
        :on-change="handleFileChange"
      >
        <el-icon class="upload-icon"><UploadFilled /></el-icon>

        <div class="upload-title">将 CSV 文件拖放到这里</div>

        <div class="upload-description">或者点击选择本地文件，建议文件大小不超过 10 MB</div>

        <el-button type="primary" plain> 选择 CSV 文件 </el-button>
      </el-upload>
    </section>

    <el-card v-if="selectedDataset || fileName" class="selected-panel" shadow="never">
      <div class="selected-information">
        <div class="selected-file-icon">
          <el-icon><Document /></el-icon>
        </div>

        <div>
          <span>当前选择</span>

          <h3>
            {{ currentDataset?.name || fileName }}
          </h3>

          <p>
            {{
              currentDataset
                ? `${currentDataset.samples} 个样本 · ${currentDataset.features} 个特征 · ${currentDataset.task}`
                : '本地上传的 CSV 数据文件'
            }}
          </p>
        </div>
      </div>

      <el-tag type="success" effect="light" round> 已准备 </el-tag>
    </el-card>

    <footer class="page-actions">
      <div>
        <strong>下一步：配置机器学习算法</strong>
        <span>选择模型并设置训练参数</span>
      </div>

      <el-button
        type="primary"
        size="large"
        :disabled="!selectedDataset && !fileName"
        @click="confirmDataset"
      >
        进入实验配置
        <el-icon class="button-icon"><ArrowRight /></el-icon>
      </el-button>
    </footer>
  </div>
</template>

<style scoped>
.data-page {
  padding-bottom: 20px;
}

.page-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 30px 34px;
  color: white;
  border-radius: 20px;
  background:
    radial-gradient(circle at 80% 10%, rgb(96 165 250 / 25%), transparent 30%),
    linear-gradient(120deg, #172554, #1e40af);
  box-shadow: 0 20px 45px rgb(30 64 175 / 14%);
}

.eyebrow {
  color: #93c5fd;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 1.8px;
}

.page-heading h1 {
  margin: 8px 0;
  font-size: 30px;
}

.page-heading p {
  margin: 0;
  color: #cbd5e1;
}

.heading-icon {
  display: grid;
  width: 72px;
  height: 72px;
  font-size: 32px;
  border: 1px solid rgb(255 255 255 / 12%);
  border-radius: 22px;
  background: rgb(255 255 255 / 9%);
  place-items: center;
}

.section {
  margin-top: 30px;
}

.section-title {
  display: flex;
  align-items: end;
  justify-content: space-between;
  margin-bottom: 17px;
}

.section-title h2 {
  margin: 0 0 6px;
  color: #0f172a;
  font-size: 21px;
}

.section-title p {
  margin: 0;
  color: #94a3b8;
  font-size: 13px;
}

.dataset-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
}

.dataset-card {
  position: relative;
  padding: 24px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 18px;
  background: white;
  box-shadow: 0 8px 25px rgb(15 23 42 / 4%);
  transition:
    transform 0.25s ease,
    border-color 0.25s ease,
    box-shadow 0.25s ease;
}

.dataset-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 18px 35px rgb(15 23 42 / 9%);
}

.dataset-card.selected {
  border-color: #3b82f6;
  box-shadow: 0 15px 35px rgb(59 130 246 / 13%);
}

.dataset-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.dataset-icon,
.selected-mark,
.selected-file-icon {
  display: grid;
  place-items: center;
}

.dataset-icon {
  width: 48px;
  height: 48px;
  font-size: 22px;
  border-radius: 14px;
}

.dataset-card.blue .dataset-icon {
  color: #2563eb;
  background: #eff6ff;
}

.dataset-card.purple .dataset-icon {
  color: #7c3aed;
  background: #f5f3ff;
}

.dataset-card.green .dataset-icon {
  color: #059669;
  background: #ecfdf5;
}

.selected-mark {
  width: 27px;
  height: 27px;
  color: white;
  border-radius: 50%;
  background: #3b82f6;
}

.dataset-card h3 {
  margin: 20px 0 9px;
  color: #0f172a;
  font-size: 18px;
}

.dataset-card > p {
  min-height: 48px;
  margin: 0;
  color: #64748b;
  font-size: 13px;
  line-height: 1.7;
}

.dataset-meta {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  padding-top: 20px;
  margin-top: 20px;
  border-top: 1px solid #eef2f7;
}

.dataset-meta div {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.dataset-meta span {
  color: #94a3b8;
  font-size: 11px;
}

.dataset-meta strong {
  color: #334155;
  font-size: 13px;
}

.divider {
  display: flex;
  align-items: center;
  gap: 20px;
  margin: 30px 0;
  color: #94a3b8;
  font-size: 12px;
}

.divider::before,
.divider::after {
  flex: 1;
  height: 1px;
  content: '';
  background: #dfe6ef;
}

.upload-section :deep(.el-upload) {
  width: 100%;
}

.upload-section :deep(.el-upload-dragger) {
  width: 100%;
  padding: 35px;
  border: 1px dashed #bfdbfe;
  border-radius: 18px;
  background: linear-gradient(135deg, #f8fbff, #f5f7ff);
}

.upload-section :deep(.el-upload-dragger:hover) {
  border-color: #3b82f6;
}

.upload-icon {
  margin-bottom: 12px;
  color: #3b82f6;
  font-size: 42px;
}

.upload-title {
  color: #1e293b;
  font-size: 16px;
  font-weight: 650;
}

.upload-description {
  margin: 8px 0 18px;
  color: #94a3b8;
  font-size: 12px;
}

.selected-panel {
  margin-top: 22px;
}

.selected-panel :deep(.el-card__body) {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.selected-information {
  display: flex;
  align-items: center;
  gap: 15px;
}

.selected-file-icon {
  width: 48px;
  height: 48px;
  color: #2563eb;
  font-size: 21px;
  border-radius: 14px;
  background: #eff6ff;
}

.selected-information span {
  color: #94a3b8;
  font-size: 11px;
}

.selected-information h3 {
  margin: 4px 0;
  color: #0f172a;
  font-size: 15px;
}

.selected-information p {
  margin: 0;
  color: #64748b;
  font-size: 12px;
}

.page-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 22px 25px;
  margin-top: 22px;
  border: 1px solid #e5eaf2;
  border-radius: 17px;
  background: white;
}

.page-actions div {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.page-actions strong {
  color: #1e293b;
  font-size: 14px;
}

.page-actions span {
  color: #94a3b8;
  font-size: 12px;
}

.button-icon {
  margin-left: 7px;
}

@media (max-width: 1000px) {
  .dataset-grid {
    grid-template-columns: 1fr;
  }

  .heading-icon {
    display: none;
  }

  .page-actions {
    align-items: stretch;
    flex-direction: column;
    gap: 18px;
  }
}
</style>
