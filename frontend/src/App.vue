<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div>
        <span class="logo">MLLab</span>
        <span class="subtitle">机器学习实验与可视化平台</span>
      </div>
      <el-tag type="success">系统运行中</el-tag>
    </el-header>

    <el-container>
      <el-aside width="220px" class="app-aside">
        <el-menu router :default-active="$route.path" class="app-menu">
          <el-menu-item index="/">
            <span>首页</span>
          </el-menu-item>

          <el-menu-item index="/data">
            <span>数据管理</span>
          </el-menu-item>

          <el-menu-item index="/experiment">
            <span>实验配置</span>
          </el-menu-item>

          <el-menu-item index="/result">
            <span>实验结果</span>
          </el-menu-item>

          <el-menu-item index="/history">
            <span>历史记录</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-container {
  min-height: 100vh;
}

.app-header {
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: #ffffff;
  border-bottom: 1px solid #e4e7ed;
}

.logo {
  color: #409eff;
  font-size: 26px;
  font-weight: bold;
}

.subtitle {
  margin-left: 16px;
  color: #606266;
  font-size: 16px;
}

.app-aside {
  min-height: calc(100vh - 64px);
  background-color: #ffffff;
  border-right: 1px solid #e4e7ed;
}

.app-menu {
  border-right: none;
}

.app-main {
  background-color: #f5f7fa;
  padding: 24px;
}
</style>
