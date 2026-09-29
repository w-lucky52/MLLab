<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Clock, Delete, Plus, Search, View } from '@element-plus/icons-vue'

const router = useRouter()

const keyword = ref('')
const selectedAlgorithm = ref('')

const experimentList = ref([
  {
    id: 'EXP-0001',
    dataset: '鸢尾花数据集',
    datasetCode: 'iris',
    algorithm: 'K 近邻（KNN）',
    algorithmCode: 'knn_classifier',
    accuracy: 94.67,
    status: '已完成',
    createdAt: '2026-09-22 18:30',
  },
  {
    id: 'EXP-0002',
    dataset: '葡萄酒数据集',
    datasetCode: 'wine',
    algorithm: '高斯朴素贝叶斯',
    algorithmCode: 'gaussian_nb',
    accuracy: 96.21,
    status: '已完成',
    createdAt: '2026-09-22 18:45',
  },
  {
    id: 'EXP-0003',
    dataset: '乳腺癌数据集',
    datasetCode: 'breast_cancer',
    algorithm: '逻辑回归',
    algorithmCode: 'logistic_regression',
    accuracy: 97.08,
    status: '已完成',
    createdAt: '2026-09-22 19:10',
  },
])

const filteredList = computed(() => {
  return experimentList.value.filter((item) => {
    const matchKeyword =
      !keyword.value ||
      item.id.toLowerCase().includes(keyword.value.toLowerCase()) ||
      item.dataset.includes(keyword.value)

    const matchAlgorithm =
      !selectedAlgorithm.value ||
      item.algorithmCode === selectedAlgorithm.value

    return matchKeyword && matchAlgorithm
  })
})

const averageAccuracy = computed(() => {
  if (!experimentList.value.length) return '0.00'

  const total = experimentList.value.reduce(
    (sum, item) => sum + item.accuracy,
    0,
  )

  return (total / experimentList.value.length).toFixed(2)
})

const clearFilters = () => {
  keyword.value = ''
  selectedAlgorithm.value = ''
}

const viewResult = (row) => {
  router.push({
    path: '/result',
    query: {
      dataset: row.datasetCode,
      algorithm: row.algorithmCode,
      testSize: 0.3,
      randomState: 42,
    },
  })
}

const deleteRecord = async (row) => {
  try {
    await ElMessageBox.confirm(
      `确定删除实验记录 ${row.id} 吗？`,
      '删除实验记录',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'warning',
      },
    )

    experimentList.value = experimentList.value.filter(
      (item) => item.id !== row.id,
    )

    ElMessage.success('实验记录已删除')
  } catch {
    // 用户取消删除时不执行操作
  }
}
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

    <el-alert
      title="当前页面展示的是模拟历史记录，连接后端后将从数据库加载真实记录。"
      type="warning"
      :closable="false"
      show-icon
      class="history-alert"
    />

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
            <strong>100%</strong>
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
  <el-option label="线性回归" value="linear_regression" />
  <el-option label="岭回归" value="ridge_regression" />
  <el-option label="决策树回归" value="decision_tree_regressor" />
  <el-option label="随机森林回归" value="random_forest_regressor" />
  <el-option label="GBDT 回归" value="gbdt_regressor" />
</el-select>
          </div>
        </div>
      </template>

      <el-table v-if="filteredList.length" :data="filteredList" class="records-table">
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

        <el-table-column label="准确率" width="130">
          <template #default="{ row }">
            <div class="accuracy-cell">
              <strong>{{ row.accuracy }}%</strong>

              <el-progress :percentage="row.accuracy" :show-text="false" :stroke-width="5" />
            </div>
          </template>
        </el-table-column>

        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag type="success" effect="light" round>
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
