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

/** Hint shown when an iterative model's history record has no stored curve. */
export const MISSING_CURVE_HINT = '该历史实验未保存训练曲线，无法展示逐轮变化。'

/**
 * Validate the stored train/test sample counts and derive percentages.
 *
 * Only the recorded counts are used — never the configured ``test_size`` —
 * so historical experiments stay accurate. Anything missing, non-integer,
 * negative, or summing to zero is unusable.
 */
export function normalizeTrainingSplit(trainCount, testCount) {
  const train = toFiniteNumber(trainCount)
  const test = toFiniteNumber(testCount)

  if (train === null || test === null) {
    return { available: false, reason: '本次实验未记录训练集与测试集样本数。' }
  }
  if (!Number.isInteger(train) || !Number.isInteger(test)) {
    return { available: false, reason: '记录的样本数不是整数。' }
  }
  if (train < 0 || test < 0) {
    return { available: false, reason: '记录的样本数为负数。' }
  }
  if (train + test === 0) {
    return { available: false, reason: '记录的样本总数为 0。' }
  }

  const total = train + test
  return {
    available: true,
    trainCount: train,
    testCount: test,
    total,
    trainPercent: (train / total) * 100,
    testPercent: (test / total) * 100,
  }
}

/**
 * Summarize a stored loss curve.
 *
 * Accepts only a non-empty one-dimensional array of finite numbers; the
 * input array is never modified. Rounds are 1-based, and a repeated minimum
 * resolves to its first occurrence. ``relativeChangePercent`` is ``null``
 * when the initial value is 0 (division by zero is never attempted).
 */
export function summarizeTrainingCurve(lossHistory) {
  if (!Array.isArray(lossHistory) || lossHistory.length === 0) {
    return { available: false, reason: '该实验未保存训练曲线。' }
  }
  if (!lossHistory.every((value) => typeof value === 'number' && Number.isFinite(value))) {
    return { available: false, reason: '训练曲线包含非有限数值。' }
  }

  const curve = [...lossHistory]
  const count = curve.length
  const initial = curve[0]
  const final = curve[count - 1]

  let minimum = curve[0]
  let minRound = 1
  for (let index = 1; index < count; index += 1) {
    if (curve[index] < minimum) {
      minimum = curve[index]
      minRound = index + 1
    }
  }

  const absoluteChange = final - initial

  return {
    available: true,
    count,
    initial,
    final,
    minimum,
    minRound,
    absoluteChange,
    relativeChangePercent:
      initial === 0 ? null : (absoluteChange / Math.abs(initial)) * 100,
    finalVsMinimum: final - minimum,
  }
}

/** Objective statements about a curve summary; never quality judgements. */
export function curveStatements(summary) {
  if (!summary?.available) return []

  const statements = []
  if (summary.absoluteChange < 0) {
    statements.push(`最终值低于初始值，绝对变化量为 ${formatNumber(summary.absoluteChange)}。`)
  } else if (summary.absoluteChange > 0) {
    statements.push(`最终值高于初始值，绝对变化量为 ${formatNumber(summary.absoluteChange)}。`)
  } else {
    statements.push('最终值等于初始值。')
  }

  statements.push(`最低记录出现在第 ${summary.minRound} 轮。`)

  if (summary.relativeChangePercent === null) {
    statements.push('初始值为 0，相对变化不适用。')
  }

  return statements
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

const parameterValueOrUnrecorded = (parameters, name) => {
  const value = parameters?.[name]
  return value === null || value === undefined ? UNRECORDED : formatParameterValue(value)
}

const MECHANISM_NOTE = '该流程为算法机制说明，不代表后端额外记录了中间状态。'

const ALGORITHM_FLOWS = {
  knn_classifier: {
    steps: [
      '保存训练样本',
      '计算待预测样本与训练样本的距离',
      '选择最近的 K 个邻居',
      '按 weights 规则投票',
      '输出类别',
    ],
    notes: (parameters) => [
      'KNN 没有基于梯度的参数优化过程，训练阶段主要保存训练数据，因此没有逐轮损失曲线。',
      `本次记录：n_neighbors = ${parameterValueOrUnrecorded(parameters, 'n_neighbors')}，weights = ${parameterValueOrUnrecorded(parameters, 'weights')}，p = ${parameterValueOrUnrecorded(parameters, 'p')}。`,
    ],
  },
  gaussian_nb: {
    steps: [
      '按类别划分训练样本',
      '估计每个类别的特征均值与方差',
      '估计类别先验',
      '计算后验相关分数',
      '输出概率最大的类别',
    ],
    notes: (parameters) => [
      'GaussianNB 通过统计估计完成训练，不执行梯度迭代，因此没有逐轮损失曲线。',
      `本次记录：var_smoothing = ${parameterValueOrUnrecorded(parameters, 'var_smoothing')}。`,
    ],
  },
  decision_tree_regressor: {
    steps: [
      '搜索候选特征与划分阈值',
      '选择使节点误差下降的划分',
      '递归生成子节点',
      '满足停止条件后生成叶节点',
      '叶节点输出预测值',
    ],
    notes: () => [
      '决策树通过递归划分构建树结构，属于结构搜索过程，不提供统一的逐轮损失曲线。',
      '树深度与节点数量是结构属性，不代表迭代轮数。',
    ],
  },
  random_forest_regressor: {
    steps: [
      '生成多组训练子样本',
      '为每组样本训练一棵决策树',
      '各棵树独立产生预测',
      '聚合多棵树的预测结果',
      '输出平均预测值',
    ],
    notes: (parameters) => [
      '随机森林由多棵决策树组成，当前接口没有记录每棵树的逐步训练损失，因此不展示统一损失曲线。',
      `本次记录：n_estimators = ${parameterValueOrUnrecorded(parameters, 'n_estimators')}，表示树的数量，不是优化迭代次数。`,
    ],
  },
  linear_regression: {
    steps: ['初始化参数', '计算预测与残差', '计算均方目标梯度', '更新参数', '检查停止条件'],
    notes: () => [MECHANISM_NOTE],
  },
  ridge_regression: {
    steps: ['初始化参数', '计算预测与残差', '加入 L2 正则项', '更新参数', '检查停止条件'],
    notes: () => [MECHANISM_NOTE],
  },
  logistic_regression: {
    steps: ['初始化类别权重', '计算 Softmax 概率', '计算交叉熵与梯度', '更新参数', '检查停止条件'],
    notes: () => [MECHANISM_NOTE],
  },
  linear_svm: {
    steps: [
      '构造 One-vs-Rest 目标',
      '计算分类间隔',
      '计算 hinge 次梯度',
      '更新参数',
      '检查停止条件',
    ],
    notes: () => [MECHANISM_NOTE],
  },
  mlp_classifier: {
    steps: ['初始化网络参数', '前向传播', '计算交叉熵', '反向传播', '更新两层网络参数'],
    notes: () => [MECHANISM_NOTE],
  },
  gbdt_regressor: {
    steps: [
      '生成初始预测',
      '计算当前残差',
      '拟合新的弱学习器',
      '更新集成预测',
      '进入下一轮提升',
    ],
    notes: () => [
      '逐轮训练分数来自 train_score_，是训练集损失，不代表测试集误差。',
      MECHANISM_NOTE,
    ],
  },
}

const GENERIC_FLOW = {
  steps: ['加载训练数据与参数', '完成模型训练', '生成预测结果'],
  notes: () => ['该模型的详细训练机制暂未提供逐步说明。'],
}

/**
 * Objective per-algorithm training mechanism. Pure description only: no
 * simulated loss data, no iteration counts invented for non-iterative
 * models, and unknown models fall back to a generic flow.
 */
export function getAlgorithmTrainingFlow(modelName, parameters) {
  const definition = ALGORITHM_FLOWS[modelName] || GENERIC_FLOW
  const safeParameters =
    parameters && typeof parameters === 'object' && !Array.isArray(parameters)
      ? parameters
      : null

  return {
    available: true,
    title: '算法训练流程',
    steps: [...definition.steps],
    notes: definition.notes(safeParameters),
  }
}

const NO_CURVE_EXPLANATIONS = {
  knn_classifier: [
    'KNN 没有基于梯度的参数优化过程。',
    '训练阶段主要保存训练数据，预测时才计算距离并完成投票。',
    '因此该模型没有逐轮损失曲线。',
  ],
  gaussian_nb: [
    'GaussianNB 通过统计估计完成训练，不执行梯度迭代。',
    '训练阶段估计类别先验与每个类别的特征分布参数。',
    '因此该模型没有逐轮损失曲线。',
  ],
  decision_tree_regressor: [
    '决策树通过递归划分构建树结构，属于结构搜索过程。',
    '训练过程中不存在统一的迭代优化目标记录。',
    '因此该模型没有逐轮损失曲线。',
  ],
  random_forest_regressor: [
    '随机森林由多棵决策树组成，各棵树独立训练。',
    '当前接口没有记录每棵树的逐步训练损失。',
    '因此不展示统一损失曲线，n_estimators 表示树的数量而非迭代次数。',
  ],
}

/** Why a model has no loss curve; unknown models get a safe fallback. */
export function getNoCurveExplanation(modelName) {
  const statements = NO_CURVE_EXPLANATIONS[modelName]
  if (statements) return [...statements]
  return ['该模型未提供逐轮损失曲线。']
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
