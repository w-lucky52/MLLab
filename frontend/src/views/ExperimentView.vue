<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

import { ArrowLeft, ArrowRight, Check, Cpu, DataAnalysis, Setting } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

const algorithm = ref('')
const testSize = ref(0.3)
const randomState = ref(42)
const algorithms = ref([])
const modelsLoading = ref(false)
const modelParameters = ref([])
const parameterValues = ref({})
const parametersLoading = ref(false)
const submitting = ref(false)

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

const modelMeta = {
  knn_classifier: {
    name: 'K 近邻分类',
    abbreviation: 'KNN',
    color: '#2563eb',
  },
  gaussian_nb: {
    name: '高斯朴素贝叶斯',
    abbreviation: 'GNB',
    color: '#7c3aed',
  },
  logistic_regression: {
    name: '逻辑回归',
    abbreviation: 'LR',
    color: '#0891b2',
  },
  linear_svm: {
    name: '线性支持向量机',
    abbreviation: 'SVM',
    color: '#db2777',
  },
  mlp_classifier: {
    name: '多层感知机',
    abbreviation: 'MLP',
    color: '#ea580c',
  },
  linear_regression: {
    name: '线性回归',
    abbreviation: 'LIN',
    color: '#16a34a',
  },
  ridge_regression: {
    name: '岭回归',
    abbreviation: 'RIDGE',
    color: '#0d9488',
  },
  decision_tree_regressor: {
    name: '决策树回归',
    abbreviation: 'CART',
    color: '#ca8a04',
  },
  random_forest_regressor: {
    name: '随机森林回归',
    abbreviation: 'RF',
    color: '#65a30d',
  },
  gbdt_regressor: {
    name: '梯度提升回归',
    abbreviation: 'GBDT',
    color: '#dc2626',
  },
}

const currentDataset = computed(() => {
  const name = route.query.dataset || 'iris'

  return {
    name,
    displayName: datasetNameMap[name] || name,
  }
})

const currentTaskType = computed(
  () => datasetTaskTypeMap[currentDataset.value.name] || 'classification',
)

const filteredAlgorithms = computed(() =>
  algorithms.value.filter((item) => item.taskType === currentTaskType.value),
)

const selectedAlgorithm = computed(() =>
  algorithms.value.find((item) => item.value === algorithm.value),
)

const fetchModels = async () => {
  modelsLoading.value = true

  try {
    const response = await fetch('/api/v1/models')

    if (!response.ok) {
      throw new Error(`模型列表请求失败：${response.status}`)
    }

    const data = await response.json()

    algorithms.value = data.map((model) => {
      const meta = modelMeta[model.name] || {}

      return {
        value: model.name,
        name: meta.name || model.name,
        abbreviation: meta.abbreviation || model.name.slice(0, 4).toUpperCase(),
        description: model.description || '暂无模型说明',
        taskType: model.task_type,
        category: model.task_type === 'classification' ? '分类模型' : '回归模型',
        color: meta.color || '#409eff',
      }
    })

    if (algorithm.value && !algorithms.value.some((item) => item.value === algorithm.value)) {
      algorithm.value = ''
    }
  } catch (error) {
    algorithms.value = []
    ElMessage.error(error.message || '无法获取算法列表，请检查后端是否启动')
  } finally {
    modelsLoading.value = false
  }
}

const selectAlgorithm = async (item) => {
  algorithm.value = item.value
  modelParameters.value = []
  parameterValues.value = {}
  parametersLoading.value = true

  try {
    const response = await fetch(`/api/v1/models/${item.value}`)

    if (!response.ok) {
      throw new Error(`模型参数请求失败：${response.status}`)
    }

    const data = await response.json()
    modelParameters.value = data.parameters || []

    parameterValues.value = Object.fromEntries(
      modelParameters.value.map((parameter) => [parameter.name, parameter.default]),
    )
  } catch (error) {
    ElMessage.error(error.message || '无法获取模型参数')
  } finally {
    parametersLoading.value = false
  }
}
const startExperiment = async () => {
  if (!algorithm.value) {
    ElMessage.warning('请先选择算法')
    return
  }

  submitting.value = true

  try {
    const response = await fetch('/api/v1/experiments', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        dataset_name: currentDataset.value.name,
        model_name: algorithm.value,
        test_size: testSize.value,
        random_state: randomState.value,
        params: parameterValues.value,
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      const message = typeof data.detail === 'string' ? data.detail : '创建实验失败'

      throw new Error(message)
    }

    ElMessage.success('实验创建成功，正在运行')

    router.push(`/result/${data.id}`)
  } catch (error) {
    ElMessage.error(error.message || '创建实验失败，请检查后端')
  } finally {
    submitting.value = false
  }
}

onMounted(fetchModels)
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
        <h3>{{ currentDataset.displayName }}</h3>
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

      <div v-loading="modelsLoading" class="algorithm-grid">
        <article
          v-for="item in filteredAlgorithms"
          :key="item.value"
          :class="['algorithm-card', item.color, { selected: algorithm === item.value }]"
          @click="selectAlgorithm(item)"
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

      <div v-if="algorithm" v-loading="parametersLoading" class="parameter-grid">
        <el-card
          v-for="parameter in modelParameters"
          :key="parameter.name"
          class="parameter-card"
          shadow="never"
        >
          <div class="parameter-info">
            <div>
              <h3>{{ parameter.label }}</h3>
              <p>{{ parameter.description || '请设置该模型参数' }}</p>
            </div>

            <el-input-number
              v-if="parameter.type === 'int' || parameter.type === 'float'"
              v-model="parameterValues[parameter.name]"
              :min="parameter.min"
              :max="parameter.max"
              :step="parameter.step ?? (parameter.type === 'int' ? 1 : 0.01)"
              controls-position="right"
            />

            <el-switch
              v-else-if="parameter.type === 'bool'"
              v-model="parameterValues[parameter.name]"
            />

            <el-select
              v-else-if="parameter.type === 'select'"
              v-model="parameterValues[parameter.name]"
              placeholder="请选择"
            >
              <el-option
                v-for="option in parameter.options || []"
                :key="option"
                :label="option"
                :value="option"
              />
            </el-select>

            <el-input v-else v-model="parameterValues[parameter.name]" />
          </div>
        </el-card>
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
        <strong>{{ currentDataset.displayName }}</strong>
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
        :loading="submitting"
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
