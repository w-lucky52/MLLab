/**
 * Pure helpers for the "训练过程" tab.
 *
 * Every helper is defensive: training records written by older versions may
 * miss any field, so functions accept null/undefined and return an explicit
 * "未记录" style result instead of NaN / undefined text.
 */

const ITERATIVE_MODEL_IDS = new Set([
  'linear_regression',
  'ridge_regression',
  'logistic_regression',
  'linear_svm',
  'mlp_classifier',
  'gbdt_regressor',
])

export const UNRECORDED = '未记录'

export function toFiniteNumber(value) {
  if (value === null || value === undefined || value === '') return null
  const result = Number(value)
  return Number.isFinite(result) ? result : null
}

export function toInteger(value) {
  const number = toFiniteNumber(value)
  return number === null ? null : Math.trunc(number)
}

const trimFixed = (value, digits) => value.toFixed(digits).replace(/\.?0+$/, '')

/** Format a finite number with at most 6 significant digits. */
export function formatNumber(value) {
  const number = toFiniteNumber(value)
  if (number === null) return UNRECORDED
  if (Number.isInteger(number)) return String(number)
  return String(Number(number.toPrecision(6)))
}

export function formatDuration(value) {
  const milliseconds = toFiniteNumber(value)
  if (milliseconds === null || milliseconds < 0) return UNRECORDED
  if (milliseconds < 1000) return `${trimFixed(milliseconds, 2)} ms`
  return `${trimFixed(milliseconds / 1000, 2)} s`
}

export function formatSampleSplit(trainingInfo) {
  const train = toInteger(trainingInfo?.train_sample_count)
  const test = toInteger(trainingInfo?.test_sample_count)
  if (train === null && test === null) return UNRECORDED

  const parts = []
  if (train !== null) parts.push(`训练 ${train}`)
  if (test !== null) parts.push(`测试 ${test}`)
  return parts.join(' / ')
}

export function iterationSummary(trainingInfo, resolvedParameters) {
  const nIter = toInteger(trainingInfo?.n_iter)
  const maxIter = toInteger(resolvedParameters?.max_iter)

  if (nIter !== null) {
    return maxIter !== null ? `${nIter} / ${maxIter}` : `${nIter} 次`
  }

  if (trainingInfo?.loss_source === 'train_score_') {
    const rounds = Array.isArray(trainingInfo?.loss_history)
      ? trainingInfo.loss_history.length
      : 0
    if (rounds > 0) return `记录 ${rounds} 轮`
  }

  return '不适用'
}

export function standardizeSummary(value) {
  if (value === true) return '已开启'
  if (value === false) return '未开启'
  return UNRECORDED
}

/** Return a finite copy of ``lossHistory``, or ``null`` when unusable. */
export function normalizeCurve(lossHistory) {
  if (!Array.isArray(lossHistory) || lossHistory.length === 0) return null

  const curve = []
  for (const item of lossHistory) {
    if (typeof item !== 'number' || !Number.isFinite(item)) return null
    curve.push(item)
  }
  return curve
}

export function curveSummary(curve) {
  if (!Array.isArray(curve) || curve.length === 0) return null

  let minimum = curve[0]
  for (const value of curve) {
    if (value < minimum) minimum = value
  }

  return {
    count: curve.length,
    initial: curve[0],
    final: curve[curve.length - 1],
    minimum,
  }
}

export function isIterativeModel(modelName) {
  return ITERATIVE_MODEL_IDS.has(modelName)
}

/**
 * Pick an integer hyperparameter for the scale summary card.
 *
 * Only recorded values are used: ``resolved_parameters`` first, then the
 * submitted ``hyperparameters`` of older experiments. Registry defaults are
 * never backfilled — anything missing shows as "未记录".
 */
const pickIntegerParameter = (resolvedParameters, hyperparameters, name) => {
  for (const source of [resolvedParameters, hyperparameters]) {
    if (!source || typeof source !== 'object' || Array.isArray(source)) continue
    const value = toInteger(source[name])
    if (value !== null) return String(value)
  }
  return UNRECORDED
}

/**
 * Third summary card of the training tab: what "model scale" means depends on
 * the algorithm. Iterative models show their iteration record; the others
 * show their most descriptive recorded parameter.
 */
export function modelScaleSummary(modelName, trainingInfo, resolvedParameters, hyperparameters) {
  if (isIterativeModel(modelName)) {
    return { label: '迭代情况', value: iterationSummary(trainingInfo, resolvedParameters) }
  }
  if (modelName === 'knn_classifier') {
    return {
      label: '邻居数量 K',
      value: pickIntegerParameter(resolvedParameters, hyperparameters, 'n_neighbors'),
    }
  }
  if (modelName === 'gaussian_nb') {
    const classes = trainingInfo?.classes
    const count = Array.isArray(classes) && classes.length > 0 ? classes.length : null
    return { label: '类别数', value: count === null ? UNRECORDED : String(count) }
  }
  if (modelName === 'decision_tree_regressor') {
    return {
      label: '最大深度',
      value: pickIntegerParameter(resolvedParameters, hyperparameters, 'max_depth'),
    }
  }
  if (modelName === 'random_forest_regressor') {
    return {
      label: '树数量',
      value: pickIntegerParameter(resolvedParameters, hyperparameters, 'n_estimators'),
    }
  }
  return { label: '模型规模', value: UNRECORDED }
}

const ALGORITHM_FLOWS = {
  knn_classifier: [
    '保存训练样本',
    '计算待预测样本与训练样本的距离',
    '选择最近的 K 个邻居',
    '按 weights 规则投票',
    '输出类别',
  ],
  gaussian_nb: [
    '按类别划分训练样本',
    '估计每个类别的特征均值与方差',
    '估计类别先验',
    '计算后验相关分数',
    '输出概率最大的类别',
  ],
  decision_tree_regressor: [
    '搜索候选特征与划分阈值',
    '选择使节点误差下降的划分',
    '递归生成子节点',
    '满足停止条件后生成叶节点',
    '叶节点输出预测值',
  ],
  random_forest_regressor: [
    '生成多组训练子样本',
    '为每组样本训练一棵决策树',
    '各棵树独立产生预测',
    '聚合多棵树的预测结果',
    '输出平均预测值',
  ],
  linear_regression: ['初始化参数', '计算预测与残差', '计算均方目标梯度', '更新参数', '检查停止条件'],
  ridge_regression: ['初始化参数', '计算预测与残差', '加入 L2 正则项', '更新参数', '检查停止条件'],
  logistic_regression: ['初始化类别权重', '计算 Softmax 概率', '计算交叉熵与梯度', '更新参数', '检查停止条件'],
  linear_svm: [
    '构造 One-vs-Rest 目标',
    '计算分类间隔',
    '计算 hinge 次梯度',
    '更新参数',
    '检查停止条件',
  ],
  mlp_classifier: ['初始化网络参数', '前向传播', '计算交叉熵', '反向传播', '更新两层网络参数'],
  gbdt_regressor: [
    '生成初始预测',
    '计算当前残差',
    '拟合新的弱学习器',
    '更新集成预测',
    '进入下一轮提升',
  ],
}

const GENERIC_FLOW = ['加载训练数据与参数', '完成模型训练', '生成预测结果']

/**
 * Objective per-algorithm training mechanism. Pure description only: no
 * simulated loss data, no iteration counts invented for non-iterative
 * models, and unknown models fall back to a generic flow.
 */
export function getAlgorithmTrainingFlow(modelName) {
  const steps = ALGORITHM_FLOWS[modelName] || GENERIC_FLOW
  return { available: true, title: '算法训练流程', steps: [...steps] }
}
