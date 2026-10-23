/**
 * Pure helpers for the "训练过程" tab.
 *
 * Every helper is defensive: training records written by older versions may
 * miss any field, so functions accept null/undefined and return an explicit
 * "未记录" style result instead of NaN / undefined text.
 */

const PARAMETER_LABELS = {
  learning_rate: '学习率',
  max_iter: '最大迭代次数',
  tol: '收敛阈值',
  alpha: '正则化系数',
  C: '正则化强度',
  n_neighbors: '邻居数量',
  weights: '权重方式',
  p: '距离度量参数',
  var_smoothing: '方差平滑系数',
  hidden_size: '隐藏层神经元数量',
  n_estimators: '基学习器数量',
  max_depth: '最大深度',
  min_samples_split: '最小分裂样本数',
  max_features: '特征采样方式',
  random_state: '随机种子',
}

const NON_ITERATIVE_NOTES = {
  knn_classifier: 'KNN 基于训练样本的邻居搜索，不进行迭代参数优化，因此没有损失曲线。',
  gaussian_nb: '高斯朴素贝叶斯通过训练数据估计类别先验及特征分布参数，不进行梯度迭代。',
  decision_tree_regressor: '决策树通过递归选择划分特征与阈值构建树结构，不提供逐轮损失曲线。',
  random_forest_regressor: '随机森林训练多棵决策树并聚合结果，不提供统一的逐轮损失曲线。',
}

const ITERATIVE_MODEL_IDS = new Set([
  'linear_regression',
  'ridge_regression',
  'logistic_regression',
  'linear_svm',
  'mlp_classifier',
  'gbdt_regressor',
])

export const UNRECORDED = '未记录'

export const LOSS_EMPTY_FALLBACK = '该历史实验未保存训练曲线，请重新运行实验后查看。'

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
  if (value === true) return { text: '已开启', hint: '' }
  if (value === false) return { text: '未开启', hint: '' }
  return {
    text: UNRECORDED,
    hint: '历史版本默认开启，但本次记录中未保存该字段。',
  }
}

export function stopReasonText(stopReason) {
  if (stopReason === 'converged') return '已满足停止条件'
  if (stopReason === 'max_iter') return '达到最大迭代次数'
  if (stopReason) return '不适用'
  return UNRECORDED
}

/** Return a finite copy of ``lossHistory``, or ``null`` when unusable. */
export function normalizeCurve(lossHistory) {
  if (!Array.isArray(lossHistory) || lossHistory.length === 0) return null

  const curve = []
  for (const item of lossHistory) {
    const value = toFiniteNumber(item)
    if (value === null) return null
    curve.push(value)
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

export function formatParameterValue(value) {
  if (value === null || value === undefined) return '空'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'number') return formatNumber(value)
  if (typeof value === 'string') return value

  try {
    return JSON.stringify(value, (key, item) => {
      if (typeof item === 'number' && !Number.isInteger(item)) {
        return Number(item.toPrecision(6))
      }
      return item
    })
  } catch {
    return String(value)
  }
}

const isNonEmptyObject = (value) =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length > 0

export function parameterRows(parameters) {
  if (!isNonEmptyObject(parameters)) return []

  return Object.keys(parameters).map((name) => ({
    name,
    label: PARAMETER_LABELS[name]
      ? `${PARAMETER_LABELS[name]}（${name}）`
      : name,
    value: formatParameterValue(parameters[name]),
  }))
}

/**
 * Decide where the parameter table comes from.
 *
 * ``resolved``  -> training_info.resolved_parameters (final, actual values)
 * ``submitted`` -> hyperparameters of an older experiment (may miss defaults)
 * ``none``      -> nothing usable was stored
 */
export function parameterSource(resolvedParameters, hyperparameters) {
  if (isNonEmptyObject(resolvedParameters)) {
    return {
      kind: 'resolved',
      title: '实际生效参数',
      note: '本次训练记录',
      warning: '',
      rows: parameterRows(resolvedParameters),
    }
  }

  if (isNonEmptyObject(hyperparameters)) {
    return {
      kind: 'submitted',
      title: '用户提交参数',
      note: '',
      warning:
        '该历史实验未保存最终生效参数，下表仅展示用户提交值，可能不包含当时的默认参数。',
      rows: parameterRows(hyperparameters),
    }
  }

  return {
    kind: 'none',
    title: '实际生效参数',
    note: '',
    warning: '',
    rows: [],
  }
}

export function isNonIterativeModel(modelName) {
  return Object.prototype.hasOwnProperty.call(NON_ITERATIVE_NOTES, modelName)
}

export function nonIterativeNote(modelName) {
  return NON_ITERATIVE_NOTES[modelName] || ''
}

export function isIterativeModel(modelName) {
  return ITERATIVE_MODEL_IDS.has(modelName)
}

export function lossEmptyDescription(modelName) {
  const note = nonIterativeNote(modelName)
  if (note) return note
  if (isIterativeModel(modelName)) return LOSS_EMPTY_FALLBACK
  return '该模型未提供逐轮损失曲线。'
}

/** Objective training-process statements, never model-quality judgements. */
export function trainingInsights({ trainingInfo, modelName, curve }) {
  const info = trainingInfo || {}
  const statements = []
  const stats = curveSummary(curve)

  if (stats) {
    statements.push(
      `训练过程共记录 ${stats.count} 个目标值，初始值为 ${formatNumber(
        stats.initial,
      )}，最终值为 ${formatNumber(stats.final)}。`,
    )
  }

  const train = toInteger(info.train_sample_count)
  const test = toInteger(info.test_sample_count)
  if (train !== null && test !== null) {
    statements.push(`本次训练使用 ${train} 个训练样本、${test} 个测试样本。`)
  }

  if (info.stop_reason === 'converged') {
    statements.push('模型在达到最大迭代次数前满足既定停止条件。')
  } else if (info.stop_reason === 'max_iter') {
    statements.push('模型运行至最大迭代次数，未触发基于 tol 的提前停止条件。')
  }

  if (info.loss_source === 'train_score_') {
    statements.push('该曲线记录梯度提升模型每一轮的训练集损失，不代表测试集误差。')
  }

  if (!stats && isNonIterativeModel(modelName)) {
    statements.push(nonIterativeNote(modelName))
  }

  if (!stats && train === null && test === null && !info.stop_reason) {
    statements.push('该历史实验未完整保存训练过程信息。')
  }

  if (statements.length === 0) {
    statements.push('该历史实验未完整保存训练过程信息。')
  }

  return statements.slice(0, 4)
}
