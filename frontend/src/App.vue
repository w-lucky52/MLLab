<script setup>
import { Clock, DataAnalysis, FolderOpened, House, Setting } from '@element-plus/icons-vue'
</script>

<template>
  <el-container class="app-shell">
    <el-aside width="248px" class="sidebar">
      <div class="brand">
        <div class="brand-icon">M</div>

        <div>
          <div class="brand-name">MLLab</div>
          <div class="brand-description">Machine Learning Lab</div>
        </div>
      </div>

      <div class="menu-label">工作台</div>

      <el-menu
        router
        :default-active="$route.path"
        class="navigation"
        background-color="transparent"
        text-color="#94a3b8"
        active-text-color="#ffffff"
      >
        <el-menu-item index="/">
          <el-icon><House /></el-icon>
          <span>概览首页</span>
        </el-menu-item>

        <el-menu-item index="/data">
          <el-icon><FolderOpened /></el-icon>
          <span>数据管理</span>
        </el-menu-item>

        <el-menu-item index="/experiment">
          <el-icon><Setting /></el-icon>
          <span>实验配置</span>
        </el-menu-item>

        <el-menu-item index="/result">
          <el-icon><DataAnalysis /></el-icon>
          <span>实验结果</span>
        </el-menu-item>

        <el-menu-item index="/history">
          <el-icon><Clock /></el-icon>
          <span>历史记录</span>
        </el-menu-item>
      </el-menu>

      <div class="sidebar-footer">
        <div class="status-dot"></div>

        <div>
          <div class="status-title">服务状态正常</div>
          <div class="status-description">MLLab v1.0</div>
        </div>
      </div>
    </el-aside>

    <el-container class="workspace">
      <el-header class="topbar">
        <div>
          <div class="page-caption">机器学习实验平台</div>
          <div class="page-title">让模型实验更简单、更直观</div>
        </div>

        <div class="topbar-actions">
          <el-tag round effect="light" type="success"> 系统运行中 </el-tag>

          <el-avatar class="avatar">ML</el-avatar>
        </div>
      </el-header>

      <el-main class="main-content">
        <div class="page-container">
          <router-view />
        </div>
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-shell {
  width: 100%;
  min-height: 100vh;
  background: #f4f7fb;
}

.sidebar {
  position: fixed;
  inset: 0 auto 0 0;
  z-index: 10;
  display: flex;
  flex-direction: column;
  padding: 0 16px 20px;
  overflow: hidden;
  background:
    radial-gradient(circle at 10% 0%, rgb(59 130 246 / 22%), transparent 32%),
    linear-gradient(180deg, #111c33 0%, #0b1220 100%);
  box-shadow: 12px 0 40px rgb(15 23 42 / 8%);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  height: 86px;
  padding: 0 8px;
}

.brand-icon {
  display: grid;
  width: 42px;
  height: 42px;
  color: white;
  font-size: 23px;
  font-weight: 800;
  place-items: center;
  border-radius: 13px;
  background: linear-gradient(135deg, #60a5fa, #6366f1);
  box-shadow: 0 10px 26px rgb(59 130 246 / 35%);
}

.brand-name {
  color: white;
  font-size: 21px;
  font-weight: 750;
  letter-spacing: 0.5px;
}

.brand-description {
  margin-top: 3px;
  color: #64748b;
  font-size: 11px;
}

.menu-label {
  padding: 20px 12px 10px;
  color: #475569;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1.8px;
}

.navigation {
  border-right: 0;
}

.navigation :deep(.el-menu-item) {
  height: 48px;
  margin: 5px 0;
  border-radius: 12px;
}

.navigation :deep(.el-menu-item:hover) {
  color: white;
  background: rgb(255 255 255 / 7%);
}

.navigation :deep(.el-menu-item.is-active) {
  background: linear-gradient(90deg, rgb(59 130 246 / 90%), rgb(99 102 241 / 88%));
  box-shadow: 0 10px 25px rgb(37 99 235 / 24%);
}

.navigation :deep(.el-icon) {
  margin-right: 12px;
  font-size: 18px;
}

.sidebar-footer {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 15px;
  margin-top: auto;
  border: 1px solid rgb(255 255 255 / 7%);
  border-radius: 14px;
  background: rgb(255 255 255 / 4%);
}

.status-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #34d399;
  box-shadow: 0 0 12px #34d399;
}

.status-title {
  color: #cbd5e1;
  font-size: 13px;
}

.status-description {
  margin-top: 3px;
  color: #64748b;
  font-size: 11px;
}

.workspace {
  min-width: 0;
  margin-left: 248px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 86px;
  padding: 0 34px;
  border-bottom: 1px solid #e9eef5;
  background: rgb(255 255 255 / 92%);
  backdrop-filter: blur(18px);
}

.page-caption {
  margin-bottom: 5px;
  color: #94a3b8;
  font-size: 12px;
}

.page-title {
  color: #1e293b;
  font-size: 17px;
  font-weight: 650;
}

.topbar-actions {
  display: flex;
  align-items: center;
  gap: 18px;
}

.avatar {
  color: white;
  font-size: 13px;
  font-weight: 700;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
}

.main-content {
  min-height: calc(100vh - 86px);
  padding: 30px 34px 44px;
}

.page-container {
  width: 100%;
  max-width: 1440px;
  margin: 0 auto;
}

@media (max-width: 800px) {
  .sidebar {
    position: static;
    width: 100% !important;
    min-height: auto;
  }

  .workspace {
    margin-left: 0;
  }

  .app-shell {
    display: block;
  }

  .sidebar-footer,
  .brand-description,
  .page-caption {
    display: none;
  }

  .main-content {
    padding: 20px;
  }
}
</style>
