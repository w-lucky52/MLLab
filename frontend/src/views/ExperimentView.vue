<script setup>
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft, ArrowRight, Check, Cpu, DataAnalysis, Setting } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

const algorithm = ref('')
const testSize = ref(0.3)
const randomState = ref(42)

const datasetNameMap = {
  iris: '鸢尾花数据集',
  wine: '葡萄酒数据集',
  breast_cancer: '乳腺癌数据集',
  digits: '手写数字数据集',
  diabetes: '糖尿病数据集',
}

const datasetTaskTypeMap = {
  iris: 'classification',
  wine: 'classification',
  breast_cancer: 'classification',
  digits: 'classification',
  diabetes: 'regression',
}

const algorithms = [
  {
    value: 'knn_classifier',
    name: 'K 近邻',
    abbreviation: 'KNN',
    description: '根据邻近样本的类别完成预测。',
    category: '分类模型',
    taskType: 'classification',
    color: 'blue',
  },
  {
    value: 'gaussian_nb',
    name: '高斯朴素贝叶斯',
    abbreviation: 'GNB',
    description: '基于概率完成分类，训练速度快。',
    category: '分类模型',
    taskType: 'classification',
    color: 'purple',
  },
  {
    value: 'logistic_regression',
    name: '逻辑回归',
    abbreviation: 'LR',
    description: '经典线性分类模型，可解释性较强。',
    category: '分类模型',
    taskType: 'classification',
    color: 'green',
  },
  {
    value: 'linear_regression',
    name: '线性回归',
    abbreviation: 'Linear',
    description: '使用线性关系预测连续数值。',
    category: '回归模型',
    taskType: 'regression',
    color: 'blue',
  },
  {
    value: 'ridge_regression',
    name: '岭回归',
    abbreviation: 'Ridge',
    description: '加入正则化的线性回归模型。',
    category: '回归模型',
    taskType: 'regression',
    color: 'purple',
  },
  {
    value: 'decision_tree_regressor',
    name: '决策树回归',
    abbreviation: 'CART',
    description: '通过树形结构预测连续数值。',
    category: '回归模型',
    taskType: 'regression',
    color: 'orange',
  },
  {
    value: 'random_forest_regressor',
    name: '随机森林回归',
    abbreviation: 'RF',
    description: '组合多棵回归树提高预测稳定性。',
    category: '回归模型',
    taskType: 'regression',
    color: 'red',
  },
  {
    value: 'gbdt_regressor',
    name: '梯度提升树回归',
    abbreviation: 'GBDT',
    description: '逐步提升模型的回归预测能力。',
    category: '回归模型',
    taskType: 'regression',
    color: 'green',
  },
]

const currentDataset = computed(() => {
  const dataset = route.query.dataset
  return datasetNameMap[dataset] || dataset || '尚未选择数据集'
})

const currentTaskType = computed(() => {
  return datasetTaskTypeMap[route.query.dataset]
})

const filteredAlgorithms = computed(() => {
  return algorithms.filter((item) => item.taskType === currentTaskType.value)
})

const selectedAlgorithm = computed(() => {
  return algorithms.find((item) => item.value === algorithm.value)
})

const startExperiment = () => {
  if (!route.query.dataset) {
    ElMessage.warning('请先选择实验数据集')
    router.push('/data')
    return
  }

  if (!algorithm.value) {
    ElMessage.warning('请选择机器学习算法')
    return
  }

  ElMessage.success('实验配置完成')

  router.push({
    path: '/result',
    query: {
      dataset: route.query.dataset,
      algorithm: algorithm.value,
      testSize: testSize.value,
      randomState: randomState.value,
    },
  })
}
</script>

<template>
  <div class="experiment-page">
    <section class="page-heading">
      <div>
        <span class="eyebrow">EXPERIMENT STUDIO</span>
        <h1>配置机器学习实验</h1>
        <p>选择合适的模型，并设置数据划分与随机参数。</p>
      </div>

      <div class="heading-icon">
        <el-icon><Setting /></el-icon>
      </div>
    </section>

    <el-alert
      v-if="!route.query.dataset"
      title="尚未选择数据集"
      description="请先进入数据管理页面选择内置数据集或上传 CSV 文件。"
      type="warning"
      :closable="false"
      show-icon
      class="dataset-warning"
    >
      <template #default>
        <el-button type="warning" size="small" @click="router.push('/data')">
          前往选择数据
        </el-button>
      </template>
    </el-alert>

    <section v-else class="dataset-summary">
      <div class="dataset-symbol">
        <el-icon><DataAnalysis /></el-icon>
      </div>

      <div>
        <span>当前实验数据</span>
        <h3>{{ currentDataset }}</h3>
      </div>

      <el-tag type="success" effect="light" round> 数据已就绪 </el-tag>

      <el-button class="change-button" link type="primary" @click="router.push('/data')">
        更换数据集
      </el-button>
    </section>

    <section class="content-section">
      <div class="section-title">
        <div>
          <span>STEP 01</span>
          <h2>选择机器学习算法</h2>
          <p>根据任务特点选择用于本次实验的模型</p>
        </div>

        <el-tag v-if="selectedAlgorithm" effect="dark" round>
          已选择 {{ selectedAlgorithm.abbreviation }}
        </el-tag>
      </div>

      <div class="algorithm-grid">
        <article
          v-for="item in filteredAlgorithms"
          :key="item.value"
          :class="['algorithm-card', item.color, { selected: algorithm === item.value }]"
          @click="algorithm = item.value"
        >
          <div class="algorithm-top">
            <div class="algorithm-icon">
              <el-icon><Cpu /></el-icon>
            </div>

            <div v-if="algorithm === item.value" class="selected-mark">
              <el-icon><Check /></el-icon>
            </div>
          </div>

          <el-tag size="small" effect="plain">
            {{ item.category }}
          </el-tag>

          <h3>
            {{ item.name }}
            <small>{{ item.abbreviation }}</small>
          </h3>

          <p>{{ item.description }}</p>
        </article>
      </div>
    </section>

    <section class="content-section">
      <div class="section-title">
        <div>
          <span>STEP 02</span>
          <h2>设置实验参数</h2>
          <p>这些参数会影响数据划分和实验复现</p>
        </div>
      </div>

      <div class="parameter-grid">
        <el-card class="parameter-card" shadow="never">
          <div class="parameter-heading">
            <div>
              <h3>测试集比例</h3>
              <p>用于模型评估的数据占全部数据的比例</p>
            </div>

            <strong>{{ Math.round(testSize * 100) }}%</strong>
          </div>

          <el-slider v-model="testSize" :min="0.2" :max="0.4" :step="0.1" :show-tooltip="false" />

          <div class="slider-labels">
            <span>20%</span>
            <span>30%</span>
            <span>40%</span>
          </div>
        </el-card>

        <el-card class="parameter-card" shadow="never">
          <div class="parameter-heading">
            <div>
              <h3>随机种子</h3>
              <p>保持相同数值可以重复获得一致的数据划分</p>
            </div>
          </div>

          <el-input-number
            v-model="randomState"
            :min="0"
            :max="9999"
            controls-position="right"
            class="number-input"
          />

          <div class="parameter-tip">推荐使用默认值 42</div>
        </el-card>
      </div>
    </section>

    <section class="configuration-summary">
      <div>
        <span>数据集</span>
        <strong>{{ currentDataset }}</strong>
      </div>

      <div>
        <span>算法</span>
        <strong>{{ selectedAlgorithm?.name || '尚未选择' }}</strong>
      </div>

      <div>
        <span>训练集 / 测试集</span>
        <strong>
          {{ Math.round((1 - testSize) * 100) }}% / {{ Math.round(testSize * 100) }}%
        </strong>
      </div>

      <div>
        <span>随机种子</span>
        <strong>{{ randomState }}</strong>
      </div>
    </section>

    <footer class="page-actions">
      <el-button size="large" @click="router.push('/data')">
        <el-icon><ArrowLeft /></el-icon>
        返回数据管理
      </el-button>

      <el-button
        type="primary"
        size="large"
        :disabled="!route.query.dataset || !algorithm"
        @click="startExperiment"
      >
        运行实验
        <el-icon class="button-icon"><ArrowRight /></el-icon>
      </el-button>
    </footer>
  </div>
</template>

<style scoped>
.experiment-page {
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
    radial-gradient(circle at 80% 10%, rgb(167 139 250 / 28%), transparent 30%),
    linear-gradient(120deg, #312e81, #4f46e5);
  box-shadow: 0 20px 45px rgb(79 70 229 / 15%);
}

.eyebrow,
.section-title span {
  color: #c4b5fd;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 1.7px;
}

.page-heading h1 {
  margin: 8px 0;
  font-size: 30px;
}

.page-heading p {
  margin: 0;
  color: #ddd6fe;
}

.heading-icon {
  display: grid;
  width: 72px;
  height: 72px;
  font-size: 32px;
  border: 1px solid rgb(255 255 255 / 13%);
  border-radius: 22px;
  background: rgb(255 255 255 / 9%);
  place-items: center;
}

.dataset-warning,
.dataset-summary {
  margin-top: 22px;
}

.dataset-summary {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 18px 22px;
  border: 1px solid #e5eaf2;
  border-radius: 16px;
  background: white;
}

.dataset-symbol,
.algorithm-icon,
.selected-mark {
  display: grid;
  place-items: center;
}

.dataset-symbol {
  width: 46px;
  height: 46px;
  color: #2563eb;
  font-size: 21px;
  border-radius: 14px;
  background: #eff6ff;
}

.dataset-summary > div:nth-child(2) {
  margin-right: auto;
}

.dataset-summary span {
  color: #94a3b8;
  font-size: 11px;
}

.dataset-summary h3 {
  margin: 4px 0 0;
  color: #1e293b;
  font-size: 15px;
}

.change-button {
  margin-left: 6px;
}

.content-section {
  margin-top: 32px;
}

.section-title {
  display: flex;
  align-items: end;
  justify-content: space-between;
  margin-bottom: 17px;
}

.section-title h2 {
  margin: 6px 0;
  color: #0f172a;
  font-size: 21px;
}

.section-title p {
  margin: 0;
  color: #94a3b8;
  font-size: 13px;
}

.algorithm-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 14px;
}

.algorithm-card {
  position: relative;
  min-height: 225px;
  padding: 19px;
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 17px;
  background: white;
  box-shadow: 0 8px 25px rgb(15 23 42 / 4%);
  transition:
    transform 0.25s ease,
    box-shadow 0.25s ease,
    border-color 0.25s ease;
}

.algorithm-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 16px 32px rgb(15 23 42 / 8%);
}

.algorithm-card.selected {
  border-color: #6366f1;
  box-shadow: 0 15px 32px rgb(99 102 241 / 12%);
}

.algorithm-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 15px;
}

.algorithm-icon {
  width: 43px;
  height: 43px;
  font-size: 20px;
  border-radius: 13px;
}

.algorithm-card.blue .algorithm-icon {
  color: #2563eb;
  background: #eff6ff;
}

.algorithm-card.purple .algorithm-icon {
  color: #7c3aed;
  background: #f5f3ff;
}

.algorithm-card.green .algorithm-icon {
  color: #059669;
  background: #ecfdf5;
}

.algorithm-card.orange .algorithm-icon {
  color: #ea580c;
  background: #fff7ed;
}

.algorithm-card.red .algorithm-icon {
  color: #dc2626;
  background: #fef2f2;
}

.selected-mark {
  width: 25px;
  height: 25px;
  color: white;
  border-radius: 50%;
  background: #6366f1;
}

.algorithm-card h3 {
  margin: 15px 0 9px;
  color: #0f172a;
  font-size: 16px;
}

.algorithm-card h3 small {
  display: block;
  margin-top: 4px;
  color: #94a3b8;
  font-size: 10px;
}

.algorithm-card p {
  margin: 0;
  color: #64748b;
  font-size: 12px;
  line-height: 1.7;
}

.parameter-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}

.parameter-card :deep(.el-card__body) {
  padding: 24px;
}

.parameter-heading {
  display: flex;
  justify-content: space-between;
  margin-bottom: 22px;
}

.parameter-heading h3 {
  margin: 0 0 7px;
  color: #1e293b;
  font-size: 16px;
}

.parameter-heading p {
  margin: 0;
  color: #94a3b8;
  font-size: 12px;
}

.parameter-heading strong {
  color: #4f46e5;
  font-size: 24px;
}

.slider-labels {
  display: flex;
  justify-content: space-between;
  color: #94a3b8;
  font-size: 10px;
}

.number-input {
  width: 100%;
}

.parameter-tip {
  margin-top: 13px;
  color: #94a3b8;
  font-size: 11px;
}

.configuration-summary {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  margin-top: 25px;
  overflow: hidden;
  border: 1px solid #e5eaf2;
  border-radius: 16px;
  background: white;
}

.configuration-summary div {
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding: 19px 22px;
  border-right: 1px solid #edf1f6;
}

.configuration-summary div:last-child {
  border-right: 0;
}

.configuration-summary span {
  color: #94a3b8;
  font-size: 11px;
}

.configuration-summary strong {
  color: #334155;
  font-size: 13px;
}

.page-actions {
  display: flex;
  justify-content: space-between;
  margin-top: 22px;
}

.button-icon {
  margin-left: 7px;
}

@media (max-width: 1200px) {
  .algorithm-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 760px) {
  .algorithm-grid,
  .parameter-grid,
  .configuration-summary {
    grid-template-columns: 1fr;
  }

  .heading-icon {
    display: none;
  }

  .configuration-summary div {
    border-right: 0;
    border-bottom: 1px solid #edf1f6;
  }
}
</style>
