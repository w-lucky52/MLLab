<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Clock, Delete, Plus, Search, View } from '@element-plus/icons-vue'

const router = useRouter()

const keyword = ref('')
const selectedAlgorithm = ref('')
const experimentList = ref([])
const loading = ref(false)

const datasetNameMap = {
  iris: '鸢尾花数据集',
  wine: '葡萄酒数据集',
  breast_cancer: '乳腺癌数据集',
  digits: '手写数字数据集',
  diabetes: '糖尿病数据集',
}

const algorithmNameMap = {
  knn_classifier: 'K 近邻（KNN）',
  gaussian_nb: '高斯朴素贝叶斯',
  logistic_regression: '逻辑回归',
  linear_svm: '线性 SVM',
  mlp_classifier: '多层感知机',
  linear_regression: '线性回归',
  ridge_regression: '岭回归',
  decision_tree_regressor: '决策树回归（CART）',
  random_forest_regressor: '随机森林回归',
  gbdt_regressor: '梯度提升树回归（GBDT）',
}

const statusNameMap = {
  pending: '等待运行',
  running: '运行中',
  completed: '已完成',
  failed: '运行失败',
}

const formatDate = (value) => {
  if (!value) return '—'

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return String(value).replace('T', ' ').slice(0, 16)
  }

  return date.toLocaleString('zh-CN', {
    hour12: false,
  })
}

const normalizeExperiment = (item) => {
  const evaluation = item.evaluation || {}
  const values = evaluation.metrics || {}
  const taskType =
    item.task_type ||
    evaluation.task_type ||
    (item.dataset_name === 'diabetes' ? 'regression' : 'classification')

  let accuracy = null
  let metricName = '—'
  let metricDisplay = '—'

  if (item.status === 'completed' && taskType === 'classification') {
    const originalAccuracy = Number(values.accuracy)

    if (Number.isFinite(originalAccuracy)) {
      accuracy = originalAccuracy <= 1 ? originalAccuracy * 100 : originalAccuracy

      metricName = '准确率'
      metricDisplay = `${accuracy.toFixed(2)}%`
    }
  }

  if (item.status === 'completed' && taskType === 'regression') {
    const r2 = Number(values.r2)

    if (Number.isFinite(r2)) {
      metricName = 'R²'
      metricDisplay = r2.toFixed(4)
    }
  }

  return {
    id: `EXP-${String(item.id).padStart(4, '0')}`,
    backendId: item.id,
    dataset: datasetNameMap[item.dataset_name] || item.dataset_name,
    datasetCode: item.dataset_name,
    algorithm: algorithmNameMap[item.model_name] || item.model_name,
    algorithmCode: item.model_name,
    taskType,
    accuracy,
    metricName,
    metricDisplay,
    status: statusNameMap[item.status] || item.status,
    statusCode: item.status,
    createdAt: formatDate(item.created_at),
  }
}

const fetchExperiments = async () => {
  loading.value = true

  try {
    const response = await fetch('/api/v1/experiments')
    const data = await response.json()

    if (!response.ok) {
      const message = typeof data.detail === 'string' ? data.detail : '获取历史记录失败'

      throw new Error(message)
    }

    experimentList.value = data.map(normalizeExperiment)
  } catch (error) {
    experimentList.value = []
    ElMessage.error(error.message || '无法获取历史记录')
  } finally {
    loading.value = false
  }
}

const filteredList = computed(() => {
  return experimentList.value.filter((item) => {
    const searchText = keyword.value.trim().toLowerCase()

    const matchKeyword =
      !searchText ||
      item.id.toLowerCase().includes(searchText) ||
      item.dataset.toLowerCase().includes(searchText) ||
      item.algorithm.toLowerCase().includes(searchText)

    const matchAlgorithm =
      !selectedAlgorithm.value || item.algorithmCode === selectedAlgorithm.value

    return matchKeyword && matchAlgorithm
  })
})

const averageAccuracy = computed(() => {
  const classificationRecords = experimentList.value.filter(
    (item) => item.taskType === 'classification' && typeof item.accuracy === 'number',
  )

  if (!classificationRecords.length) return '0.00'

  const total = classificationRecords.reduce((sum, item) => sum + item.accuracy, 0)

  return (total / classificationRecords.length).toFixed(2)
})

const completionRate = computed(() => {
  if (!experimentList.value.length) return '0.00'

  const completedCount = experimentList.value.filter(
    (item) => item.statusCode === 'completed',
  ).length

  return ((completedCount / experimentList.value.length) * 100).toFixed(2)
})

const statusTagType = (status) => {
  return (
    {
      pending: 'info',
      running: 'warning',
      completed: 'success',
      failed: 'danger',
    }[status] || 'info'
  )
}
const clearFilters = () => {
  keyword.value = ''
  selectedAlgorithm.value = ''
}

const viewResult = (row) => {
  if (row.statusCode !== 'completed') {
    ElMessage.warning('该实验尚未完成，暂时无法查看结果')
    return
  }

  router.push(`/result/${row.backendId}`)
}

const deleteRecord = async (row) => {
  try {
    await ElMessageBox.confirm(`确定删除实验记录 ${row.id} 吗？`, '删除实验记录', {
      confirmButtonText: '确定删除',
      cancelButtonText: '取消',
      type: 'warning',
    })

    const response = await fetch(`/api/v1/experiments/${row.backendId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      let message = '删除实验记录失败'

      try {
        const data = await response.json()

        if (typeof data.detail === 'string') {
          message = data.detail
        }
      } catch {
        // 204 等无响应体情况无需处理
      }

      throw new Error(message)
    }

    experimentList.value = experimentList.value.filter((item) => item.backendId !== row.backendId)

    ElMessage.success('实验记录已删除')
  } catch (error) {
    if (error === 'cancel' || error === 'close') return

    ElMessage.error(error.message || '删除实验记录失败')
  }
}

onMounted(fetchExperiments)
</script>

<template>
  <div class="history-page">
    <section class="page-heading">
      <div>
        <span class="eyebrow">EXPERIMENT ARCHIVE</span>
        <h1>历史实验记录</h1>
        <p>管理、查找和比较以前运行的机器学习实验。</p>
      </div>

      <el-button type="primary" size="large" @click="router.push('/data')">
        <el-icon><Plus /></el-icon>
        新建实验
      </el-button>
    </section>

    <section class="statistics-grid">
      <el-card shadow="never">
        <div class="statistic">
          <div class="statistic-icon blue">
            <el-icon><Clock /></el-icon>
          </div>

          <div>
            <span>实验总数</span>
            <strong>{{ experimentList.length }}</strong>
            <small>累计实验记录</small>
          </div>
        </div>
      </el-card>

      <el-card shadow="never">
        <div class="statistic">
          <div class="statistic-icon green">
            <span>%</span>
          </div>

          <div>
            <span>平均准确率</span>
            <strong>{{ averageAccuracy }}%</strong>
            <small>所有实验平均值</small>
          </div>
        </div>
      </el-card>

      <el-card shadow="never">
        <div class="statistic">
          <div class="statistic-icon purple">
            <span>✓</span>
          </div>

          <div>
            <span>完成率</span>
            <strong>{{ completionRate }}%</strong>
            <small>实验运行状态</small>
          </div>
        </div>
      </el-card>
    </section>

    <el-card class="records-card" shadow="never">
      <template #header>
        <div class="records-header">
          <div>
            <h2>实验列表</h2>
            <p>共找到 {{ filteredList.length }} 条实验记录</p>
          </div>

          <div class="filters">
            <el-input
              v-model="keyword"
              placeholder="搜索编号或数据集"
              clearable
              class="search-input"
            >
              <template #prefix>
                <el-icon><Search /></el-icon>
              </template>
            </el-input>

            <el-select
              v-model="selectedAlgorithm"
              placeholder="全部算法"
              clearable
              class="algorithm-select"
            >
              <el-option label="K 近邻" value="knn_classifier" />
              <el-option label="高斯朴素贝叶斯" value="gaussian_nb" />
              <el-option label="逻辑回归" value="logistic_regression" />
              <el-option label="线性 SVM" value="linear_svm" />
              <el-option label="多层感知机" value="mlp_classifier" />
              <el-option label="线性回归" value="linear_regression" />
              <el-option label="岭回归" value="ridge_regression" />
              <el-option label="决策树回归" value="decision_tree_regressor" />
              <el-option label="随机森林回归" value="random_forest_regressor" />
              <el-option label="GBDT 回归" value="gbdt_regressor" />
            </el-select>
          </div>
        </div>
      </template>

      <el-table
        v-if="filteredList.length"
        v-loading="loading"
        :data="filteredList"
        class="records-table"
      >
        <el-table-column label="实验编号" min-width="130">
          <template #default="{ row }">
            <strong class="experiment-id">{{ row.id }}</strong>
          </template>
        </el-table-column>

        <el-table-column label="数据集" min-width="175">
          <template #default="{ row }">
            <div class="dataset-cell">
              <div class="dataset-dot"></div>
              <span>{{ row.dataset }}</span>
            </div>
          </template>
        </el-table-column>

        <el-table-column prop="algorithm" label="机器学习算法" min-width="180" />

        <el-table-column label="核心指标" width="130">
          <template #default="{ row }">
            <div class="accuracy-cell">
              <strong>{{ row.metricName }}：{{ row.metricDisplay }}</strong>

              <el-progress
                v-if="typeof row.accuracy === 'number'"
                :percentage="row.accuracy"
                :show-text="false"
                :stroke-width="5"
              />
            </div>
          </template>
        </el-table-column>

        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.statusCode)" effect="light" round>
              {{ row.status }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column prop="createdAt" label="运行时间" min-width="170" />

        <el-table-column label="操作" width="145" fixed="right">
          <template #default="{ row }">
            <el-tooltip content="查看实验结果">
              <el-button circle type="primary" plain @click="viewResult(row)">
                <el-icon><View /></el-icon>
              </el-button>
            </el-tooltip>

            <el-tooltip content="删除实验记录">
              <el-button circle type="danger" plain @click="deleteRecord(row)">
                <el-icon><Delete /></el-icon>
              </el-button>
            </el-tooltip>
          </template>
        </el-table-column>
      </el-table>

      <el-empty v-else description="没有找到符合条件的实验记录">
        <el-button v-if="keyword || selectedAlgorithm" @click="clearFilters">
          清除筛选条件
        </el-button>

        <el-button v-else type="primary" @click="router.push('/data')"> 开始第一次实验 </el-button>
      </el-empty>
    </el-card>
  </div>
</template>

<style scoped>
.history-page {
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
    radial-gradient(circle at 82% 10%, rgb(192 132 252 / 25%), transparent 30%),
    linear-gradient(120deg, #3b0764, #6d28d9);
  box-shadow: 0 20px 45px rgb(109 40 217 / 14%);
}

.eyebrow {
  color: #d8b4fe;
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
  color: #e9d5ff;
}

.history-alert {
  margin-top: 22px;
  border-radius: 13px;
}

.statistics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin-top: 20px;
}

.statistic {
  display: flex;
  align-items: center;
  gap: 16px;
}

.statistic-icon {
  display: grid;
  flex: none;
  width: 52px;
  height: 52px;
  font-size: 22px;
  font-weight: 750;
  border-radius: 15px;
  place-items: center;
}

.statistic-icon.blue {
  color: #2563eb;
  background: #eff6ff;
}

.statistic-icon.green {
  color: #059669;
  background: #ecfdf5;
}

.statistic-icon.purple {
  color: #7c3aed;
  background: #f5f3ff;
}

.statistic > div:last-child {
  display: grid;
  grid-template-columns: 1fr auto;
  flex: 1;
  align-items: center;
}

.statistic span {
  color: #64748b;
  font-size: 12px;
}

.statistic strong {
  grid-row: span 2;
  color: #0f172a;
  font-size: 25px;
}

.statistic small {
  margin-top: 5px;
  color: #94a3b8;
}

.records-card {
  margin-top: 20px;
}

.records-card :deep(.el-card__header) {
  padding: 20px 22px;
}

.records-card :deep(.el-card__body) {
  padding: 0 22px 20px;
}

.records-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.records-header h2 {
  margin: 0 0 5px;
  color: #1e293b;
  font-size: 17px;
}

.records-header p {
  margin: 0;
  color: #94a3b8;
  font-size: 11px;
}

.filters {
  display: flex;
  gap: 12px;
}

.search-input {
  width: 230px;
}

.algorithm-select {
  width: 180px;
}

.records-table {
  width: 100%;
}

.records-table :deep(th.el-table__cell) {
  color: #64748b;
  font-size: 11px;
  font-weight: 650;
  background: #f8fafc;
}

.records-table :deep(td.el-table__cell) {
  padding: 15px 0;
}

.experiment-id {
  color: #2563eb;
  font-size: 12px;
}

.dataset-cell {
  display: flex;
  align-items: center;
  gap: 9px;
}

.dataset-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #6366f1;
  box-shadow: 0 0 8px rgb(99 102 241 / 40%);
}

.accuracy-cell {
  width: 85px;
}

.accuracy-cell strong {
  display: block;
  margin-bottom: 7px;
  color: #334155;
  font-size: 12px;
}

@media (max-width: 900px) {
  .statistics-grid {
    grid-template-columns: 1fr;
  }

  .records-header {
    align-items: stretch;
    flex-direction: column;
    gap: 15px;
  }

  .filters {
    flex-direction: column;
  }

  .search-input,
  .algorithm-select {
    width: 100%;
  }
}
</style>
