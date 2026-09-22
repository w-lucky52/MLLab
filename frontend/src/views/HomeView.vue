<script setup>
import { useRouter } from 'vue-router'
import {
  ArrowRight,
  DataAnalysis,
  FolderOpened,
  MagicStick,
  TrendCharts,
} from '@element-plus/icons-vue'

const router = useRouter()

const statistics = [
  {
    label: '内置数据集',
    value: '3',
    description: '支持上传 CSV',
    icon: FolderOpened,
    color: 'blue',
  },
  {
    label: '可选算法',
    value: '5',
    description: '覆盖分类与集成模型',
    icon: MagicStick,
    color: 'purple',
  },
  {
    label: '评估指标',
    value: '4',
    description: '准确率、精确率等',
    icon: TrendCharts,
    color: 'green',
  },
]

const features = [
  {
    title: '数据管理',
    description: '选择内置数据集，或上传自己的 CSV 数据文件。',
    path: '/data',
    icon: FolderOpened,
  },
  {
    title: '实验配置',
    description: '选择机器学习算法，灵活调整实验参数。',
    path: '/experiment',
    icon: MagicStick,
  },
  {
    title: '结果分析',
    description: '通过指标和图表直观分析模型表现。',
    path: '/result',
    icon: DataAnalysis,
  },
]
</script>

<template>
  <div class="home-page">
    <section class="hero">
      <div class="hero-decoration decoration-one"></div>
      <div class="hero-decoration decoration-two"></div>

      <div class="hero-content">
        <div class="hero-badge">
          <span class="badge-dot"></span>
          Machine Learning Laboratory
        </div>

        <h1>
          从数据到模型，
          <span>让实验一目了然</span>
        </h1>

        <p>
          一站式完成数据选择、算法配置、模型训练和结果分析， 帮助你更高效地理解机器学习实验过程。
        </p>

        <div class="hero-actions">
          <el-button type="primary" size="large" @click="router.push('/data')">
            开始新实验
            <el-icon class="button-icon"><ArrowRight /></el-icon>
          </el-button>

          <el-button size="large" @click="router.push('/history')"> 查看历史记录 </el-button>
        </div>
      </div>

      <div class="model-panel">
        <div class="panel-header">
          <span>模型运行状态</span>
          <el-tag type="success" effect="dark" round>Ready</el-tag>
        </div>

        <div class="model-graphic">
          <div class="graphic-ring ring-one"></div>
          <div class="graphic-ring ring-two"></div>
          <div class="graphic-center">ML</div>
        </div>

        <div class="panel-information">
          <span>实验环境</span>
          <strong>已准备就绪</strong>
        </div>
      </div>
    </section>

    <section class="statistics-grid">
      <el-card v-for="item in statistics" :key="item.label" class="statistic-card" shadow="never">
        <div :class="['statistic-icon', item.color]">
          <el-icon><component :is="item.icon" /></el-icon>
        </div>

        <div class="statistic-information">
          <span>{{ item.label }}</span>
          <strong>{{ item.value }}</strong>
          <small>{{ item.description }}</small>
        </div>
      </el-card>
    </section>

    <section class="content-section">
      <div class="section-heading">
        <div>
          <span class="section-caption">CORE FEATURES</span>
          <h2>从这里开始你的实验</h2>
        </div>

        <el-button link type="primary" @click="router.push('/data')">
          进入数据管理
          <el-icon><ArrowRight /></el-icon>
        </el-button>
      </div>

      <div class="feature-grid">
        <el-card
          v-for="(item, index) in features"
          :key="item.path"
          class="feature-card"
          shadow="never"
          @click="router.push(item.path)"
        >
          <div class="feature-top">
            <div class="feature-icon">
              <el-icon><component :is="item.icon" /></el-icon>
            </div>

            <span>0{{ index + 1 }}</span>
          </div>

          <h3>{{ item.title }}</h3>
          <p>{{ item.description }}</p>

          <div class="feature-link">
            立即使用
            <el-icon><ArrowRight /></el-icon>
          </div>
        </el-card>
      </div>
    </section>

    <el-card class="workflow-card" shadow="never">
      <div class="workflow-heading">
        <span>STANDARD WORKFLOW</span>
        <h2>四步完成机器学习实验</h2>
      </div>

      <el-steps :active="0" align-center>
        <el-step title="选择数据" description="选择或上传数据集" />
        <el-step title="设置参数" description="选择算法与参数" />
        <el-step title="运行实验" description="训练并评估模型" />
        <el-step title="分析结果" description="查看指标与图表" />
      </el-steps>
    </el-card>
  </div>
</template>

<style scoped>
.home-page {
  padding-bottom: 20px;
}

.hero {
  position: relative;
  display: grid;
  grid-template-columns: 1.45fr 0.55fr;
  min-height: 390px;
  padding: 54px 58px;
  overflow: hidden;
  color: white;
  border-radius: 24px;
  background: linear-gradient(120deg, #172554 0%, #1e3a8a 52%, #4338ca 100%);
  box-shadow: 0 25px 60px rgb(30 64 175 / 18%);
}

.hero-content {
  position: relative;
  z-index: 2;
  align-self: center;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  padding: 7px 13px;
  color: #bfdbfe;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1.2px;
  border: 1px solid rgb(255 255 255 / 14%);
  border-radius: 20px;
  background: rgb(255 255 255 / 8%);
}

.badge-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #60a5fa;
  box-shadow: 0 0 10px #60a5fa;
}

.hero h1 {
  max-width: 650px;
  margin: 24px 0 17px;
  font-size: clamp(38px, 4vw, 57px);
  line-height: 1.18;
  letter-spacing: -2px;
}

.hero h1 span {
  color: #93c5fd;
}

.hero p {
  max-width: 610px;
  margin: 0 0 30px;
  color: #cbd5e1;
  font-size: 16px;
  line-height: 1.9;
}

.hero-actions {
  display: flex;
  gap: 12px;
}

.button-icon {
  margin-left: 7px;
}

.model-panel {
  position: relative;
  z-index: 2;
  align-self: center;
  padding: 22px;
  border: 1px solid rgb(255 255 255 / 14%);
  border-radius: 20px;
  background: rgb(255 255 255 / 8%);
  box-shadow: 0 20px 40px rgb(0 0 0 / 14%);
  backdrop-filter: blur(18px);
}

.panel-header,
.panel-information {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.panel-header {
  color: #cbd5e1;
  font-size: 13px;
}

.model-graphic {
  position: relative;
  display: grid;
  width: 150px;
  height: 150px;
  margin: 30px auto;
  place-items: center;
}

.graphic-ring {
  position: absolute;
  border: 1px solid rgb(147 197 253 / 35%);
  border-radius: 50%;
}

.ring-one {
  width: 150px;
  height: 150px;
}

.ring-two {
  width: 105px;
  height: 105px;
}

.graphic-center {
  display: grid;
  width: 65px;
  height: 65px;
  font-size: 20px;
  font-weight: 800;
  border-radius: 20px;
  place-items: center;
  background: linear-gradient(135deg, #60a5fa, #818cf8);
  box-shadow: 0 0 35px rgb(96 165 250 / 45%);
}

.panel-information {
  padding-top: 17px;
  color: #94a3b8;
  font-size: 12px;
  border-top: 1px solid rgb(255 255 255 / 10%);
}

.panel-information strong {
  color: #ffffff;
}

.hero-decoration {
  position: absolute;
  border-radius: 50%;
  filter: blur(5px);
}

.decoration-one {
  top: -150px;
  right: 16%;
  width: 330px;
  height: 330px;
  background: rgb(59 130 246 / 18%);
}

.decoration-two {
  right: -90px;
  bottom: -170px;
  width: 350px;
  height: 350px;
  background: rgb(129 140 248 / 20%);
}

.statistics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin-top: 22px;
}

.statistic-card :deep(.el-card__body) {
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 22px;
}

.statistic-icon,
.feature-icon {
  display: grid;
  flex: none;
  place-items: center;
}

.statistic-icon {
  width: 52px;
  height: 52px;
  font-size: 23px;
  border-radius: 15px;
}

.statistic-icon.blue {
  color: #2563eb;
  background: #eff6ff;
}

.statistic-icon.purple {
  color: #7c3aed;
  background: #f5f3ff;
}

.statistic-icon.green {
  color: #059669;
  background: #ecfdf5;
}

.statistic-information {
  display: grid;
  grid-template-columns: 1fr auto;
  flex: 1;
  align-items: center;
}

.statistic-information span {
  color: #64748b;
  font-size: 13px;
}

.statistic-information strong {
  grid-row: span 2;
  color: #0f172a;
  font-size: 30px;
}

.statistic-information small {
  margin-top: 5px;
  color: #94a3b8;
}

.content-section {
  margin-top: 34px;
}

.section-heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  margin-bottom: 18px;
}

.section-caption,
.workflow-heading span {
  color: #3b82f6;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 1.6px;
}

.section-heading h2,
.workflow-heading h2 {
  margin: 6px 0 0;
  color: #0f172a;
  font-size: 24px;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
}

.feature-card {
  cursor: pointer;
  transition:
    transform 0.25s ease,
    box-shadow 0.25s ease;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 18px 35px rgb(15 23 42 / 8%);
}

.feature-card :deep(.el-card__body) {
  padding: 25px;
}

.feature-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.feature-top > span {
  color: #dbe3ef;
  font-size: 26px;
  font-weight: 800;
}

.feature-icon {
  width: 46px;
  height: 46px;
  color: #2563eb;
  font-size: 21px;
  border-radius: 14px;
  background: #eff6ff;
}

.feature-card h3 {
  margin: 22px 0 9px;
  color: #0f172a;
  font-size: 18px;
}

.feature-card p {
  min-height: 50px;
  margin: 0;
  color: #64748b;
  font-size: 14px;
  line-height: 1.7;
}

.feature-link {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-top: 19px;
  color: #2563eb;
  font-size: 13px;
  font-weight: 650;
}

.workflow-card {
  margin-top: 26px;
}

.workflow-card :deep(.el-card__body) {
  padding: 28px 34px 34px;
}

.workflow-heading {
  margin-bottom: 30px;
}

@media (max-width: 1000px) {
  .hero {
    grid-template-columns: 1fr;
  }

  .model-panel {
    display: none;
  }

  .statistics-grid,
  .feature-grid {
    grid-template-columns: 1fr;
  }
}
</style>
