<script setup>
import { Clock } from '@element-plus/icons-vue'

defineProps({
  title: { type: String, required: true },
  description: { type: String, required: true },
})
</script>

<template>
  <el-card class="placeholder-card" shadow="never">
    <div class="placeholder">
      <div class="placeholder-icon">
        <el-icon><Clock /></el-icon>
      </div>

      <div>
        <h3>{{ title }}</h3>
        <p>{{ description }}</p>
      </div>
    </div>
  </el-card>
</template>

<style scoped>
.placeholder-card {
  margin-top: 18px;
}

.placeholder-card :deep(.el-card__body) {
  padding: 26px 24px;
}

.placeholder {
  display: flex;
  align-items: center;
  gap: 16px;
}

.placeholder-icon {
  display: grid;
  width: 46px;
  height: 46px;
  flex: none;
  color: #2563eb;
  font-size: 21px;
  border-radius: 14px;
  background: #eff6ff;
  place-items: center;
}

.placeholder h3 {
  margin: 0 0 6px;
  color: #1e293b;
  font-size: 15px;
}

.placeholder p {
  margin: 0;
  color: #64748b;
  font-size: 12px;
  line-height: 1.7;
}
</style>
