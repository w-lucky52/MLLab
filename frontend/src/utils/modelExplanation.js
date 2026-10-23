/**
 * Pure helpers for the "模型解释" tab.
 *
 * The backend stores only what a model can faithfully expose. These helpers
 * re-validate every array defensively and return ``null`` when a value is
 * missing, non-finite or shape-mismatched. Nothing is truncated, zero-padded
 * or re-ordered silently, and no business feature name is ever guessed.
 */

import { UNRECORDED, formatNumber, toFiniteNumber, toInteger } from './training'

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

/** Build display rows from a per-class intercept vector, or ``null``. */
export function interceptEntries(intercept, classNames) {
  const count = classNames?.names?.length ?? 0
  if (!Array.isArray(intercept) || count === 0 || intercept.length !== count) return null

  const entries = []
  for (let index = 0; index < count; index += 1) {
    const value = intercept[index]
    if (!isFiniteNumber(value)) return null
    entries.push({ label: classNames.names[index], value })
  }
  return entries
}

/**
 * Build signed, magnitude-sorted coefficient rows.
 *
 * Sorting happens on a copy; the stored coefficient itself is never replaced
 * by its absolute value.
 */
export function coefficientRows(vector, featureNames) {
  const values = normalizeVector(vector)
  if (!values) return null

  const names = normalizeFeatureNames(featureNames, values.length)
  return values
    .map((value, index) => ({
      feature: names.names[index],
      value,
      magnitude: Math.abs(value),
      direction: value > 0 ? 'positive' : value < 0 ? 'negative' : 'zero',
    }))
    .sort((a, b) => b.magnitude - a.magnitude)
}

/** Everything the regression coefficient view needs, or ``null``. */
export function regressionCoefficientSummary({ coefficients, featureNames, intercept }) {
  const rows = coefficientRows(coefficients, featureNames)
  if (!rows) return null

  const values = normalizeVector(coefficients)
  const names = normalizeFeatureNames(featureNames, values.length)
  const top = rows[0] || null

  return {
    rows,
    featureNames: names,
    featureCount: values.length,
    intercept: normalizeIntercept(intercept),
    topFeature: top ? top.feature : null,
    topMagnitude: top ? top.magnitude : null,
  }
}

/**
 * Build descending importance rows plus a legal percentage view.
 *
 * Percentages are computed only when every stored value is non-negative and
 * the total is positive. Raw values are always kept unchanged.
 */
export function importanceSummary({ featureImportances, featureNames }) {
  const values = normalizeVector(featureImportances)
  if (!values) return null

  const names = normalizeFeatureNames(featureNames, values.length)
  let total = 0
  let percentAvailable = true
  for (const value of values) {
    if (value < 0) percentAvailable = false
    total += value
  }
  if (!(total > 0)) percentAvailable = false

  const percents = percentAvailable ? values.map((value) => (value / total) * 100) : null
  const rows = values
    .map((value, index) => ({
      feature: names.names[index],
      value,
      percent: percents ? percents[index] : null,
    }))
    .sort((a, b) => b.value - a.value)

  const top3Share = percentAvailable
    ? rows.slice(0, 3).reduce((sum, row) => sum + row.value, 0) / total * 100
    : null

  return {
    rows,
    featureNames: names,
    featureCount: values.length,
    topFeature: rows[0]?.feature ?? null,
    topValue: rows[0]?.value ?? null,
    percentAvailable,
    top3Share,
    total: percentAvailable ? total : null,
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

/** Layer dimensions and the parameter total for the MLP view. */
export function mlpDimensions(trainingInfo) {
  const info = trainingInfo || {}
  const resolved = isPlainObject(info.resolved_parameters) ? info.resolved_parameters : {}
  const input = toInteger(info.n_features_in)
  const hidden = toInteger(resolved.hidden_size)
  const output = Array.isArray(info.classes) && info.classes.length > 0 ? info.classes.length : null

  const positive = (value) => Number.isInteger(value) && value > 0
  const valid = positive(input) && positive(hidden) && positive(output)
  const parameterCount = valid
    ? input * hidden + hidden + hidden * output + output
    : null

  return {
    input: positive(input) ? input : null,
    hidden: positive(hidden) ? hidden : null,
    output: positive(output) ? output : null,
    valid,
    parameterCount,
    inputText: positive(input) ? String(input) : UNRECORDED,
    hiddenText: positive(hidden) ? String(hidden) : UNRECORDED,
    outputText: positive(output) ? String(output) : UNRECORDED,
    parameterText: parameterCount === null ? UNRECORDED : String(parameterCount),
    breakdown: valid
      ? [
          { label: '输入→隐藏 权重', value: input * hidden },
          { label: '隐藏层偏置', value: hidden },
          { label: '隐藏→输出 权重', value: hidden * output },
          { label: '输出层偏置', value: output },
        ]
      : [],
  }
}

const weightsText = (weights) => {
  if (weights === 'uniform') return '等权投票（uniform）'
  if (weights === 'distance') return '距离越近权重越大（distance）'
  if (typeof weights === 'string' && weights.trim()) return weights
  return UNRECORDED
}

const distanceText = (p) => {
  const value = toFiniteNumber(p)
  if (value === null) return UNRECORDED
  if (value === 1) return '曼哈顿距离（p=1）'
  if (value === 2) return '欧氏距离（p=2）'
  if (value > 0) return `Minkowski 距离（p=${formatNumber(value)}）`
  return UNRECORDED
}

/** Mechanism summary for the KNN view. */
export function knnExplanationData(trainingInfo) {
  const info = trainingInfo || {}
  const resolved = isPlainObject(info.resolved_parameters) ? info.resolved_parameters : {}
  const neighbors = toInteger(resolved.n_neighbors)
  const trainSamples = toInteger(info.train_sample_count)
  const weights = resolved.weights ?? null
  const p = resolved.p ?? null

  return {
    neighborsText: neighbors !== null && neighbors > 0 ? String(neighbors) : UNRECORDED,
    weightsText: weightsText(weights),
    distanceText: distanceText(p),
    trainSamplesText:
      trainSamples !== null && trainSamples > 0 ? String(trainSamples) : UNRECORDED,
    workflow: [
      '计算待预测样本与已保存训练样本之间的距离。',
      `选择距离最近的 K 个样本（K = ${neighbors !== null && neighbors > 0 ? neighbors : UNRECORDED}）。`,
      `按权重策略汇总邻居投票：${weightsText(weights)}。`,
    ],
  }
}

/** Objective interpretation notes. Never model-quality or causal claims. */
export function explanationNotes(kind, trainingInfo) {
  const info = trainingInfo || {}
  const standardized = info.standardize === true

  if (kind === 'linear-regression') {
    const notes = [
      '正系数表示该特征增大时预测值倾向增加，负系数表示预测值倾向降低。',
      '系数反映当前训练数据与特征尺度下的线性关系，不代表变量之间存在因果关系。',
    ]
    if (standardized) {
      notes.push('本次训练特征已标准化，系数基于标准化后的特征尺度，不同特征之间可直接比较大小。')
    } else {
      notes.push('系数大小会受特征自身量纲影响，不同量纲的特征不宜直接比较。')
    }
    return notes
  }

  if (kind === 'linear-classifier') {
    return [
      '系数描述该特征对对应类别 score 的影响方向与幅度，不等同于概率变化。',
      '系数绝对值会受特征尺度影响，不能据此认定某个特征在现实中起决定作用。',
      standardized
        ? '本次训练特征已标准化，系数基于标准化后的特征尺度。'
        : '不同特征量纲不同，系数大小不宜直接比较。',
    ]
  }

  if (kind === 'mlp') {
    return [
      '本页面展示网络结构，不把隐藏层连接权重解释为可靠的全局特征重要性。',
      '网络结构本身不提供可直接归因到单个特征的重要性分数。',
    ]
  }

  if (kind === 'knn') {
    return [
      'KNN 不学习全局系数或特征重要性，其预测取决于局部邻域。',
      '邻居数量、权重策略与距离度量共同决定分类结果。',
    ]
  }

  if (kind === 'gaussian-nb') {
    return [
      '均值描述某类别中特征的中心位置，方差描述该特征在类别内部的离散程度。',
      '类别之间的统计差异不代表因果关系。',
    ]
  }

  if (kind === 'tree') {
    return [
      '特征重要性表示该模型在训练过程中使用特征进行划分的相对贡献，不代表因果关系。',
      '不同树模型的重要性计算方式可能不同，数值不宜跨模型直接比较。',
    ]
  }

  return []
}
