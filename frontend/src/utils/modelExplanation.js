/**
 * Pure helpers for the "模型解释" tab.
 *
 * The backend stores only what a model can faithfully expose. These helpers
 * re-validate every array defensively and return ``null`` when a value is
 * missing, non-finite or shape-mismatched. Nothing is truncated, zero-padded
 * or re-ordered silently, and no business feature name is ever guessed.
 */

import { UNRECORDED, formatNumber, toInteger } from './training.js'

export const POSITIONAL_FEATURE_PREFIX = '特征'
export const POSITIONAL_CLASS_PREFIX = '类别'

const isPlainObject = (value) =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

const isFiniteNumber = (value) => typeof value === 'number' && Number.isFinite(value)

const positionalFeatures = (count) =>
  Array.from({ length: count }, (_, index) => `${POSITIONAL_FEATURE_PREFIX} ${index + 1}`)

const positionalClasses = (count) =>
  Array.from({ length: count }, (_, index) => `${POSITIONAL_CLASS_PREFIX} ${index + 1}`)

/** Map a registered model id to the explanation view it supports. */
export function explanationKind(modelName) {
  if (modelName === 'linear_regression' || modelName === 'ridge_regression') {
    return 'linear-regression'
  }
  if (modelName === 'logistic_regression' || modelName === 'linear_svm') {
    return 'linear-classifier'
  }
  if (modelName === 'mlp_classifier') return 'mlp'
  if (modelName === 'knn_classifier') return 'knn'
  if (modelName === 'gaussian_nb') return 'gaussian-nb'
  if (
    modelName === 'decision_tree_regressor' ||
    modelName === 'random_forest_regressor' ||
    modelName === 'gbdt_regressor'
  ) {
    return 'tree'
  }
  return 'unknown'
}

/**
 * Resolve feature labels for ``columnCount`` columns.
 *
 * Real names are used only when the stored array is a non-empty list of
 * non-empty strings with exactly ``columnCount`` items. Anything else falls
 * back to positional labels without ever inventing a business name.
 */
export function normalizeFeatureNames(featureNames, columnCount) {
  const count = toInteger(columnCount)
  if (count === null || count <= 0) return { names: [], positional: true, mismatched: false }

  const stored = Array.isArray(featureNames)
    ? featureNames.filter((name) => typeof name === 'string' && name.trim().length > 0)
    : null

  if (stored && Array.isArray(featureNames) && featureNames.length === count && stored.length === count) {
    return {
      names: stored.map((name) => name.trim()),
      positional: false,
      mismatched: false,
    }
  }

  return {
    names: positionalFeatures(count),
    positional: true,
    mismatched: Boolean(Array.isArray(featureNames) && featureNames.length > 0),
  }
}

/** One-line note explaining where the feature labels came from. */
export function featureNameNotice(featureNames) {
  if (!featureNames?.positional) return ''
  if (featureNames.mismatched) {
    return '特征名称与数据列数不一致，以下使用位置标签（特征 1、特征 2……）。'
  }
  return '该实验未记录特征名称，以下使用位置标签（特征 1、特征 2……）。'
}

const classLabel = (value) => {
  if (typeof value === 'string') return value.trim() ? value : null
  if (isFiniteNumber(value)) return formatNumber(value)
  return null
}

const readClassLabels = (values, rowCount) => {
  if (!Array.isArray(values) || values.length !== rowCount) return null
  const labels = []
  for (const value of values) {
    const label = classLabel(value)
    if (label === null) return null
    labels.push(label)
  }
  return labels.length > 0 ? labels : null
}

/**
 * Resolve row labels for a "class × feature" matrix.
 *
 * Priority: training_info.classes -> evaluation score_classes -> positional
 * labels. ``source`` tells the page which origin was used so fallback labels
 * are never presented as recorded class names.
 */
export function normalizeClassNames(classes, scoreClasses, rowCount) {
  const count = toInteger(rowCount)
  if (count === null || count <= 0) {
    return { names: [], source: 'positional', fallback: true }
  }

  const fromTraining = readClassLabels(classes, count)
  if (fromTraining) return { names: fromTraining, source: 'training', fallback: false }

  const fromScores = readClassLabels(scoreClasses, count)
  if (fromScores) return { names: fromScores, source: 'score', fallback: true }

  return { names: positionalClasses(count), source: 'positional', fallback: true }
}

/** Note explaining a fallback class-label source, empty for recorded labels. */
export function classNameNotice(classNames) {
  if (!classNames?.fallback) return ''
  if (classNames.source === 'score') {
    return '该实验未在训练信息中记录类别顺序，类别名称取自评估结果（score_classes）。'
  }
  return '该实验未记录类别名称，以下使用位置标签（类别 1、类别 2……），不代表真实类别名称。'
}

/** Return a finite non-empty 1-D numeric array, or ``null``. */
export function normalizeVector(values) {
  if (!Array.isArray(values) || values.length === 0) return null
  const vector = []
  for (const value of values) {
    if (!isFiniteNumber(value)) return null
    vector.push(value)
  }
  return vector
}

/** Return a finite non-empty rectangular matrix, or ``null``. */
export function normalizeMatrix(values) {
  if (!Array.isArray(values) || values.length === 0) return null

  const rows = []
  let columnCount = null
  for (const row of values) {
    if (!Array.isArray(row) || row.length === 0) return null
    if (columnCount === null) columnCount = row.length
    else if (columnCount !== row.length) return null

    const converted = []
    for (const value of row) {
      if (!isFiniteNumber(value)) return null
      converted.push(value)
    }
    rows.push(converted)
  }
  if (columnCount === null || columnCount <= 0) return null
  return { rows, rowCount: rows.length, columnCount }
}

/** Normalize the intercept to a single finite number, or ``null``. */
export function normalizeIntercept(intercept) {
  if (isFiniteNumber(intercept)) return intercept
  if (Array.isArray(intercept) && intercept.length === 1 && isFiniteNumber(intercept[0])) {
    return intercept[0]
  }
  return null
}

/**
 * 线性回归/岭回归系数数据。
 *
 * 输入非法时返回 ``available: false`` 与明确原因；排序稳定（绝对值相同时
 * 保持原始特征顺序）；不修改输入数组；原始系数始终带符号保留。
 */
export function summarizeCoefficients({ coefficients, featureNames, intercept } = {}) {
  const values = normalizeVector(coefficients)
  if (!values) {
    return {
      available: false,
      reason: '该实验未记录可用的特征系数，或系数包含非有限数值。',
    }
  }

  const names = normalizeFeatureNames(featureNames, values.length)

  const rows = values
    .map((value, index) => ({
      feature: names.names[index],
      value,
      magnitude: Math.abs(value),
      direction: value > 0 ? 'positive' : value < 0 ? 'negative' : 'zero',
    }))
    .sort((left, right) => right.magnitude - left.magnitude)

  return {
    available: true,
    reason: '',
    rows,
    featureNames: names,
    featureCount: values.length,
    intercept: normalizeIntercept(intercept),
  }
}

/**
 * A per-class intercept vector usable as a heatmap column, or ``null``.
 *
 * Only an array with exactly ``rowCount`` finite entries qualifies; anything
 * else (single number, wrong length, non-finite values) is left out.
 */
export function interceptColumn(intercept, rowCount) {
  const count = toInteger(rowCount)
  if (!Array.isArray(intercept) || count === null || count <= 0) return null
  if (intercept.length !== count) return null

  const column = []
  for (const value of intercept) {
    if (!isFiniteNumber(value)) return null
    column.push(value)
  }
  return column
}

/**
 * 逻辑回归/线性 SVM 的类别 × 特征系数数据。
 *
 * 类别与特征的行列顺序完全保持后端记录的顺序，不合并、不重排。矩阵不规则
 * 或含非有限值时整块不可用。截距形状与类别数一致时作为 ``heatmapMatrix``
 * 的最后一列（逐行复制，不修改输入），否则不追加。
 */
export function summarizeMulticlassCoefficients({
  coefficients,
  featureNames,
  classes,
  scoreClasses,
  intercept,
} = {}) {
  const matrix = normalizeMatrix(coefficients)
  if (!matrix) {
    return {
      available: false,
      reason: '该实验未记录可用的类别系数矩阵，或矩阵形状不合法。',
    }
  }

  const names = normalizeFeatureNames(featureNames, matrix.columnCount)
  const classNames = normalizeClassNames(classes, scoreClasses, matrix.rowCount)

  const extraColumn = interceptColumn(intercept, matrix.rowCount)
  const heatmapMatrix = extraColumn === null
    ? matrix.rows
    : matrix.rows.map((row, index) => [...row, extraColumn[index]])

  return {
    available: true,
    reason: '',
    classCount: matrix.rowCount,
    featureCount: matrix.columnCount,
    classNames,
    featureNames: names,
    matrix: matrix.rows,
    heatmapMatrix,
    interceptAppended: extraColumn !== null,
  }
}

/**
 * 树模型特征重要性数据。
 *
 * 负数重要性直接不可用；特征名称数量与重要性数量不一致时也不可用。
 * 排名按重要性降序，同值保持原始特征顺序（稳定排序）；百分比仅在总和
 * 大于 0 时计算，原始重要性与派生百分比同时保留。
 */
export function summarizeFeatureImportances({ featureImportances, featureNames } = {}) {
  const values = normalizeVector(featureImportances)
  if (!values) {
    return {
      available: false,
      reason: '该实验未记录可用的特征重要性，或数值包含非有限值。',
    }
  }
  if (values.some((value) => value < 0)) {
    return { available: false, reason: '特征重要性包含负数，无法生成可靠的摘要。' }
  }

  const storedNames = Array.isArray(featureNames)
    ? featureNames.filter((name) => typeof name === 'string' && name.trim().length > 0)
    : []
  if (storedNames.length > 0 && storedNames.length !== values.length) {
    return { available: false, reason: '特征名称数量与重要性数量不一致，无法生成可靠的摘要。' }
  }

  const names = normalizeFeatureNames(featureNames, values.length)
  const total = values.reduce((sum, value) => sum + value, 0)
  const percentAvailable = total > 0

  const rows = values
    .map((value, index) => ({
      feature: names.names[index],
      value,
      percent: percentAvailable ? (value / total) * 100 : null,
    }))
    .sort((left, right) => right.value - left.value)

  return {
    available: true,
    reason: '',
    rows,
    featureNames: names,
    featureCount: values.length,
  }
}

/**
 * Validate the GaussianNB mean/variance pair.
 *
 * Both matrices must exist with the same shape; row count comes from the
 * recorded classes and column count from the feature labels. Variance must be
 * non-negative. Returns ``null`` when the pair cannot be rendered faithfully.
 */
export function gaussianExplanationData({ trainingInfo, featureNames, scoreClasses }) {
  const info = trainingInfo || {}
  const means = normalizeMatrix(info.class_feature_means)
  const variances = normalizeMatrix(info.class_feature_variances)
  if (!means || !variances) return null
  if (means.rowCount !== variances.rowCount) return null
  if (means.columnCount !== variances.columnCount) return null

  let varianceValuesLegal = true
  for (const row of variances.rows) {
    for (const value of row) {
      if (value < 0) varianceValuesLegal = false
    }
  }
  if (!varianceValuesLegal) return null

  const names = normalizeFeatureNames(featureNames, means.columnCount)
  const classes = normalizeClassNames(info.classes, scoreClasses, means.rowCount)

  return {
    means: means.rows,
    variances: variances.rows,
    rowCount: means.rowCount,
    columnCount: means.columnCount,
    featureNames: names,
    classes,
  }
}

/** Collapse a matrix into the ``[rowIndex, columnIndex, value]`` heatmap form. */
export function heatmapCells(matrix) {
  const data = normalizeMatrix(matrix)
  if (!data) return null

  const cells = []
  data.rows.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      cells.push([columnIndex, rowIndex, value])
    })
  })
  return { cells, rowCount: data.rowCount, columnCount: data.columnCount }
}

/** Symmetric color range for signed values. */
export function symmetricRange(matrix) {
  const data = normalizeMatrix(matrix)
  if (!data) return null

  let maxAbs = 0
  for (const row of data.rows) {
    for (const value of row) {
      const magnitude = Math.abs(value)
      if (magnitude > maxAbs) maxAbs = magnitude
    }
  }
  return { min: -maxAbs, max: maxAbs === 0 ? 1 : maxAbs, maxAbs }
}

/** Non-negative color range for variance-like values, or ``null``. */
export function nonNegativeRange(matrix) {
  const data = normalizeMatrix(matrix)
  if (!data) return null

  let maximum = 0
  for (const row of data.rows) {
    for (const value of row) {
      if (value < 0) return null
      if (value > maximum) maximum = value
    }
  }
  return { min: 0, max: maximum === 0 ? 1 : maximum, maximum }
}

/** Layer dimensions for the MLP view; missing values stay "未记录". */
export function mlpDimensions(trainingInfo) {
  const info = trainingInfo || {}
  const resolved = isPlainObject(info.resolved_parameters) ? info.resolved_parameters : {}
  const input = toInteger(info.n_features_in)
  const hidden = toInteger(resolved.hidden_size)
  const output = Array.isArray(info.classes) && info.classes.length > 0 ? info.classes.length : null

  const positive = (value) => Number.isInteger(value) && value > 0

  return {
    input: positive(input) ? input : null,
    hidden: positive(hidden) ? hidden : null,
    output: positive(output) ? output : null,
    inputText: positive(input) ? String(input) : UNRECORDED,
    hiddenText: positive(hidden) ? String(hidden) : UNRECORDED,
    outputText: positive(output) ? String(output) : UNRECORDED,
  }
}

/** MLP 网络结构摘要：仅展示记录的三个维度。 */
export function summarizeMLPArchitecture(trainingInfo) {
  const dims = mlpDimensions(trainingInfo)
  return {
    available: true,
    reason: '',
    inputText: dims.inputText,
    hiddenText: dims.hiddenText,
    outputText: dims.outputText,
    structureText: `${dims.inputText} → ${dims.hiddenText} → ${dims.outputText}`,
  }
}

const weightsText = (weights) => {
  if (weights === 'uniform') return '等权投票（uniform）'
  if (weights === 'distance') return '距离越近权重越大（distance）'
  if (typeof weights === 'string' && weights.trim()) return weights
  return UNRECORDED
}

/** Mechanism workflow for the KNN view. */
export function knnExplanationData(trainingInfo) {
  const info = trainingInfo || {}
  const resolved = isPlainObject(info.resolved_parameters) ? info.resolved_parameters : {}
  const neighbors = toInteger(resolved.n_neighbors)
  const weights = resolved.weights ?? null

  return {
    workflow: [
      '计算待预测样本与已保存训练样本之间的距离。',
      `选择距离最近的 K 个样本（K = ${neighbors !== null && neighbors > 0 ? neighbors : UNRECORDED}）。`,
      `按权重策略汇总邻居投票：${weightsText(weights)}。`,
    ],
  }
}

// --- Phase 7B2B: real per-model explanation payloads --------------------------
//
// The backend stores the real payloads under ``training_info.explanation_data``.
// The helpers below re-validate every array defensively, never mutate their
// input, keep the recorded order unless a view explicitly re-sorts (stably),
// and return ``available: false`` instead of fabricating or repairing values.

const KNOWN_EXPLANATION_KINDS = new Set([
  'linear_contribution',
  'multiclass_linear_contribution',
  'knn_neighbors',
  'gaussian_nb_likelihood',
  'decision_path',
  'permutation_importance',
])

// Kinds whose payload carries per-sample local explanations.
const SAMPLE_KINDS = new Set([
  'linear_contribution',
  'multiclass_linear_contribution',
  'knn_neighbors',
  'gaussian_nb_likelihood',
  'decision_path',
])

// Reconstruction tolerance for the header integrity check (the backend
// already enforces 1e-6; the display check stays one order looser).
const RECONSTRUCTION_RTOL = 1e-4
const RECONSTRUCTION_ATOL = 1e-4

const isClose = (actual, expected) =>
  Math.abs(actual - expected) <=
  RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * Math.abs(expected)

const isPositiveInteger = (value) => Number.isInteger(value) && value > 0

const isValidSample = (sample) =>
  isPlainObject(sample) && isPositiveInteger(sample.sample_index)

/**
 * Validate the stored ``training_info.explanation_data`` payload.
 *
 * Only payloads with a known kind, the recorded standardized feature space and
 * a consistent feature-name list qualify. Sample-based kinds must carry at
 * least one sample with a positive integer ``sample_index``; anything else is
 * reported as ``available: false`` so the page falls back to the global view
 * instead of rendering a partial local card.
 */
export function normalizeExplanationData(raw) {
  const unavailable = { available: false, kind: '', featureNames: [], samples: null, raw: null }
  if (!isPlainObject(raw)) return unavailable

  const kind = typeof raw.kind === 'string' ? raw.kind : ''
  if (!KNOWN_EXPLANATION_KINDS.has(kind)) return unavailable
  if (raw.feature_space !== 'standardized') return unavailable

  const names = Array.isArray(raw.feature_names)
    ? raw.feature_names.filter((name) => typeof name === 'string' && name.trim().length > 0)
    : []
  if (names.length === 0 || names.length !== raw.feature_names.length) return unavailable

  let samples = null
  if (SAMPLE_KINDS.has(kind)) {
    if (!Array.isArray(raw.samples) || raw.samples.length === 0) return unavailable
    if (!raw.samples.every(isValidSample)) return unavailable
    samples = raw.samples
  }

  return {
    available: true,
    kind,
    featureNames: names.map((name) => name.trim()),
    featureSpace: raw.feature_space,
    samples,
    scoreName: typeof raw.score_name === 'string' ? raw.score_name : null,
    classes: Array.isArray(raw.classes) ? raw.classes : null,
    raw,
  }
}

/** The validated sample list of a normalized payload (``[]`` when absent). */
export function explanationSamples(explanation) {
  if (!explanation?.available || !Array.isArray(explanation.samples)) return []
  return explanation.samples
}

const readFiniteVector = (values, expectedLength) => {
  if (!Array.isArray(values) || values.length !== expectedLength) return null
  if (!values.every(isFiniteNumber)) return null
  return values
}

/**
 * Per-feature signed contributions of one test sample (linear / ridge
 * regression and the predicted class of logistic regression / linear SVM).
 *
 * Rows are sorted by absolute contribution, descending, with a stable sort so
 * ties keep the original feature order. ``sum(contributions) + intercept`` is
 * checked against the stored reconstructed value; a mismatch makes the whole
 * sample unavailable instead of displaying inconsistent numbers.
 */
export function contributionChartData({ sample, featureNames } = {}) {
  const unavailable = { available: false, rows: [] }
  if (!isValidSample(sample)) return unavailable
  const names = Array.isArray(featureNames) ? featureNames : []
  if (names.length === 0) return unavailable

  const contributions = readFiniteVector(sample.contributions, names.length)
  const featureValues = readFiniteVector(sample.feature_values, names.length)
  const intercept = isFiniteNumber(sample.intercept) ? sample.intercept : null
  const reconstructedValue = [sample.reconstructed_prediction, sample.reconstructed_score].find(
    isFiniteNumber,
  )
  if (!contributions || !featureValues || intercept === null || reconstructedValue === undefined) {
    return unavailable
  }

  const contributionSum = contributions.reduce((sum, value) => sum + value, 0)
  if (!isClose(contributionSum + intercept, reconstructedValue)) return unavailable

  const rows = contributions
    .map((contribution, index) => ({
      feature: names[index],
      featureValue: featureValues[index],
      contribution,
      magnitude: Math.abs(contribution),
      direction: contribution > 0 ? 'positive' : contribution < 0 ? 'negative' : 'zero',
    }))
    .sort((left, right) => right.magnitude - left.magnitude)

  return {
    available: true,
    rows,
    intercept,
    contributionSum,
    reconstructed: reconstructedValue,
    sampleIndex: sample.sample_index,
    yTrue: sample.y_true ?? null,
    yPred: sample.y_pred ?? null,
  }
}

/**
 * Neighbour rows of one KNN test sample, in the stored rank order.
 *
 * ``trainIndex`` is the position inside the training set (never a raw dataset
 * row number); distances and weights are the recorded values, not recomputed.
 */
export function knnNeighborChartData(sample) {
  const unavailable = { available: false, rows: [] }
  if (!isValidSample(sample)) return unavailable
  const neighbors = sample.neighbors
  if (!Array.isArray(neighbors) || neighbors.length === 0) return unavailable

  const rows = []
  let lastRank = 0
  for (const neighbor of neighbors) {
    if (!isPlainObject(neighbor)) return unavailable
    if (!isPositiveInteger(neighbor.rank) || neighbor.rank <= lastRank) return unavailable
    if (!Number.isInteger(neighbor.train_index) || neighbor.train_index < 0) return unavailable
    if (!isFiniteNumber(neighbor.distance) || neighbor.distance < 0) return unavailable
    if (!isFiniteNumber(neighbor.vote_weight) || neighbor.vote_weight < 0) return unavailable
    lastRank = neighbor.rank
    rows.push({
      rank: neighbor.rank,
      name: `邻居 #${neighbor.rank}`,
      trainIndex: neighbor.train_index,
      distance: neighbor.distance,
      weight: neighbor.vote_weight,
      classValue: neighbor.label ?? null,
    })
  }
  return { available: true, rows }
}

/**
 * Class vote totals of one KNN test sample, in the recorded class order.
 *
 * The stored ``vote_totals`` are used as-is; the page never re-derives votes.
 */
export function knnVoteChartData(sample) {
  const unavailable = { available: false, rows: [], predicted: null }
  if (!isValidSample(sample)) return unavailable
  const totals = sample.vote_totals
  if (!Array.isArray(totals) || totals.length === 0) return unavailable

  const rows = []
  for (const entry of totals) {
    if (!isPlainObject(entry)) return unavailable
    if (!isFiniteNumber(entry.weight) || entry.weight < 0) return unavailable
    if (entry.class === null || entry.class === undefined) return unavailable
    rows.push({ classValue: entry.class, weight: entry.weight })
  }
  return { available: true, rows, predicted: sample.y_pred ?? null }
}

/**
 * Feature log-likelihoods of one GaussianNB test sample, in feature order.
 *
 * ``log_prior + sum(feature_log_likelihoods)`` is checked against the stored
 * joint log likelihood; the values keep their sign and are never re-labelled
 * as probability contributions.
 */
export function gaussianLikelihoodChartData({ sample, featureNames } = {}) {
  const unavailable = { available: false, rows: [] }
  if (!isValidSample(sample)) return unavailable
  const names = Array.isArray(featureNames) ? featureNames : []
  if (names.length === 0) return unavailable

  const likelihoods = readFiniteVector(sample.feature_log_likelihoods, names.length)
  const featureValues = readFiniteVector(sample.feature_values, names.length)
  const logPrior = isFiniteNumber(sample.log_prior) ? sample.log_prior : null
  const joint = isFiniteNumber(sample.joint_log_likelihood)
    ? sample.joint_log_likelihood
    : null
  if (!likelihoods || !featureValues || logPrior === null || joint === null) {
    return unavailable
  }
  const rebuilt = logPrior + likelihoods.reduce((sum, value) => sum + value, 0)
  if (!isClose(rebuilt, joint)) return unavailable

  let predictedProbability = null
  if (Array.isArray(sample.class_probabilities)) {
    const hit = sample.class_probabilities.find(
      (entry) =>
        isPlainObject(entry) &&
        String(entry.class) === String(sample.predicted_class) &&
        isFiniteNumber(entry.probability),
    )
    if (hit) predictedProbability = hit.probability
  }

  const rows = likelihoods.map((logLikelihood, index) => ({
    feature: names[index],
    featureValue: featureValues[index],
    logLikelihood,
  }))

  return {
    available: true,
    rows,
    predictedClass: sample.predicted_class ?? null,
    logPrior,
    joint,
    predictedProbability,
    sampleIndex: sample.sample_index,
  }
}

/**
 * Ordered decision-path steps of one CART test sample.
 *
 * The stored root-to-leaf order is kept as-is; each row carries a ready-made
 * Chinese sentence and the direction is checked against the recorded
 * comparison (left ⇔ value ≤ threshold). The leaf prediction is returned
 * separately and never re-derived.
 */
export function decisionPathRows(sample) {
  const unavailable = { available: false, steps: [], leafPrediction: null, leafText: '' }
  if (!isValidSample(sample)) return unavailable
  if (!Array.isArray(sample.path)) return unavailable
  if (!isFiniteNumber(sample.leaf_prediction)) return unavailable

  const steps = []
  let lastStep = 0
  for (const node of sample.path) {
    if (!isPlainObject(node)) return unavailable
    if (!isPositiveInteger(node.step) || node.step <= lastStep) return unavailable
    if (typeof node.feature_name !== 'string' || !node.feature_name.trim()) return unavailable
    if (!isFiniteNumber(node.value) || !isFiniteNumber(node.threshold)) return unavailable
    if (node.direction !== 'left' && node.direction !== 'right') return unavailable
    const goesLeft = node.value <= node.threshold
    if ((node.direction === 'left') !== goesLeft) return unavailable
    lastStep = node.step

    const comparison = goesLeft ? '≤' : '>'
    const branch = goesLeft ? '左分支' : '右分支'
    steps.push({
      step: node.step,
      featureName: node.feature_name.trim(),
      value: node.value,
      threshold: node.threshold,
      direction: node.direction,
      text: `第 ${node.step} 步：${node.feature_name.trim()} = ${formatNumber(node.value)} ${comparison} ${formatNumber(node.threshold)}，进入${branch}`,
    })
  }

  const leafPrediction = sample.leaf_prediction
  return {
    available: true,
    steps,
    leafPrediction,
    leafText: `叶节点预测值：${formatNumber(leafPrediction)}`,
  }
}

/**
 * Test-set permutation importance rows (MLP accuracy drop / ensemble MSE
 * increase).
 *
 * Negative importances are kept; rows are sorted by value descending with a
 * stable sort, and the input payload is never modified. The metric identity
 * comes from the stored baseline field, never from the model name.
 */
export function permutationImportanceData(raw) {
  const unavailable = { available: false, rows: [] }
  if (!isPlainObject(raw) || raw.kind !== 'permutation_importance') return unavailable

  let metric = null
  let baseline = null
  if (isFiniteNumber(raw.baseline_accuracy)) {
    metric = 'accuracy'
    baseline = raw.baseline_accuracy
  } else if (isFiniteNumber(raw.baseline_mse)) {
    metric = 'mse'
    baseline = raw.baseline_mse
  }
  if (metric === null) return unavailable

  const names = Array.isArray(raw.feature_names)
    ? raw.feature_names.filter((name) => typeof name === 'string' && name.trim().length > 0)
    : []
  const importances = Array.isArray(raw.importances) ? raw.importances : null
  if (!importances || names.length === 0 || names.length !== raw.feature_names.length) {
    return unavailable
  }
  if (importances.length !== names.length || !importances.every(isFiniteNumber)) {
    return unavailable
  }

  const rows = importances
    .map((importance, index) => ({
      feature: names[index].trim(),
      importance,
      direction: importance > 0 ? 'positive' : importance < 0 ? 'negative' : 'zero',
    }))
    .sort((left, right) => right.importance - left.importance)

  return {
    available: true,
    rows,
    metric,
    baseline,
    baselineLabel: metric === 'accuracy' ? '基准准确率' : '基准 MSE',
    deltaLabel: metric === 'accuracy' ? '准确率变化' : 'MSE 增量',
    metricTag: metric === 'accuracy' ? '准确率下降' : 'MSE 增量',
    importanceName: typeof raw.importance_name === 'string' ? raw.importance_name : '',
    featureCount: names.length,
  }
}
