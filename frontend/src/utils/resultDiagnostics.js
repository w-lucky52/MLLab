/**
 * Pure diagnostics helpers for the result page.
 *
 * Everything here is derived from the values already stored on the experiment
 * (``y_true`` / ``y_pred`` / ``y_score``). Nothing is simulated, reordered or
 * invented, and every function leaves its inputs untouched.
 *
 * Functions return an object with ``available: false`` plus a human readable
 * ``reason`` when the data cannot be used, so the Vue components can render a
 * compact empty state instead of an empty axis.
 */

import { resolveClassDisplayNames } from './classDisplay.js'

const isFiniteNumber = (value) => typeof value === 'number' && Number.isFinite(value)

const isScalarLabel = (value) =>
  (typeof value === 'string' && value.length > 0) || isFiniteNumber(value)

/** Turn a flat array of labels into ``{key, value}`` pairs, or ``null``. */
const toLabelItems = (values) => {
  if (!Array.isArray(values) || values.length === 0) return null
  const items = []
  for (const value of values) {
    if (!isScalarLabel(value)) return null
    items.push({
      // Numeric and string labels never collapse into the same bucket.
      key: typeof value === 'number' ? `n:${value}` : `s:${value}`,
      value,
    })
  }
  return items
}

const unavailable = (reason) => ({ available: false, reason })

const formatValue = (value) => {
  if (!Number.isFinite(value)) return '—'
  return String(Number(value.toFixed(4)))
}

/**
 * Count actual vs predicted samples per class.
 *
 * Class order: ``scoreClasses`` first (the backend order), then any label that
 * only appears in ``y_true`` / ``y_pred`` in first-appearance order. Labels are
 * never sorted automatically.
 */
export function classDistribution(yTrue, yPred, scoreClasses, scoreClassNames) {
  if (!Array.isArray(yTrue) || !Array.isArray(yPred)) {
    return unavailable('缺少真实类别或预测类别数据。')
  }
  if (yTrue.length === 0 || yPred.length === 0) {
    return unavailable('该实验未保存真实类别或预测类别数据。')
  }
  if (yTrue.length !== yPred.length) {
    return unavailable('真实类别与预测类别的样本数量不一致。')
  }

  const actualItems = toLabelItems(yTrue)
  const predictedItems = toLabelItems(yPred)
  if (!actualItems || !predictedItems) {
    return unavailable('类别标签包含不支持或非有限的值。')
  }

  const scoreItems = Array.isArray(scoreClasses) ? toLabelItems(scoreClasses) : null

  const order = []
  const seen = new Set()
  const push = (item) => {
    if (seen.has(item.key)) return
    seen.add(item.key)
    order.push(item)
  }

  if (scoreItems) scoreItems.forEach(push)
  actualItems.forEach(push)
  predictedItems.forEach(push)

  const counts = new Map(order.map((item) => [item.key, { actual: 0, predicted: 0 }]))
  actualItems.forEach((item) => {
    counts.get(item.key).actual += 1
  })
  predictedItems.forEach((item) => {
    counts.get(item.key).predicted += 1
  })

  const nameByKey = new Map()
  if (scoreItems) {
    const displayNames = resolveClassDisplayNames(scoreClasses, scoreClassNames)
    scoreItems.forEach((item, index) => {
      nameByKey.set(item.key, displayNames[index] ?? String(item.value))
    })
  }

  const classes = order.map((item) => ({
    key: item.key,
    value: item.value,
    name: nameByKey.get(item.key) ?? String(item.value),
    actualCount: counts.get(item.key).actual,
    predictedCount: counts.get(item.key).predicted,
  }))

  return {
    available: true,
    classes,
    actualTotal: yTrue.length,
    predictedTotal: yPred.length,
  }
}

const CONFIDENCE_TITLES = {
  probability: {
    title: '预测置信度分布',
    description: '置信度为模型对最终预测类别给出的最大概率，不代表概率一定经过校准。',
  },
  decision: {
    title: '决策间隔分布',
    description: '决策间隔表示得分最高类别与次高类别之间的差值，不是概率。',
  },
}

/**
 * Per-sample confidence (probability) or decision margin.
 *
 * ``probability``: ``max(row)`` of a finite row in [0, 1] that sums to 1.
 * ``decision``: ``top1 - top2`` of the raw decision scores (never a
 * probability). Ties produce a margin of exactly 0.
 */
export function predictionConfidence(yTrue, yPred, yScore, scoreType) {
  if (!Array.isArray(yTrue) || !Array.isArray(yPred)) {
    return unavailable('缺少真实标签或预测标签数据。')
  }
  if (yTrue.length === 0 || yPred.length === 0 || yTrue.length !== yPred.length) {
    return unavailable('真实标签与预测标签的样本数量不一致。')
  }
  if (!Array.isArray(yScore) || yScore.length === 0) {
    return unavailable('该实验未保存预测分数，无法计算置信度。')
  }
  if (yScore.length !== yTrue.length) {
    return unavailable('预测分数的行数与样本数量不一致。')
  }

  const kind = scoreType === 'decision' ? 'decision' : 'probability'
  let columnCount = null
  for (const row of yScore) {
    if (!Array.isArray(row) || row.length === 0 || !row.every(isFiniteNumber)) {
      return unavailable('预测分数包含不支持或非有限的值。')
    }
    if (columnCount === null) columnCount = row.length
    else if (row.length !== columnCount) {
      return unavailable('预测分数的列数不一致。')
    }
  }

  const values = []
  const correct = []

  for (let index = 0; index < yScore.length; index += 1) {
    const row = yScore[index]
    correct.push(String(yTrue[index]) === String(yPred[index]))

    if (kind === 'probability') {
      if (!row.every((value) => value >= 0 && value <= 1)) {
        return unavailable('预测概率必须位于 0 与 1 之间。')
      }
      const total = row.reduce((sum, value) => sum + value, 0)
      if (Math.abs(total - 1) > 1e-6) {
        return unavailable('预测概率每行之和应当为 1。')
      }
      values.push(Math.max(...row))
    } else {
      if (row.length < 2) {
        return unavailable('决策分数至少需要两个类别才能计算决策间隔。')
      }
      const sorted = [...row].sort((left, right) => right - left)
      const margin = sorted[0] - sorted[1]
      if (!Number.isFinite(margin) || margin < 0) {
        return unavailable('决策间隔必须为非负有限数。')
      }
      values.push(margin)
    }
  }

  const meta = CONFIDENCE_TITLES[kind]
  const correctCount = correct.filter(Boolean).length

  return {
    available: true,
    kind,
    title: meta.title,
    description: meta.description,
    values,
    correct,
    correctCount,
    wrongCount: correct.length - correctCount,
  }
}

/**
 * Deterministic equal-width histogram.
 *
 * The last bin owns the maximum value so no sample is dropped; a constant
 * array still produces one valid bin. ``flags`` (optional) marks samples as
 * correct/wrong for the stacked confidence chart.
 */
export function buildBins(values, binCount = 10, flags = null) {
  if (!Array.isArray(values) || values.length === 0) {
    return { available: false, reason: '没有可用于统计的数值。', bins: [], total: 0 }
  }
  if (!values.every(isFiniteNumber)) {
    return {
      available: false,
      reason: '数值包含非有限值，无法生成直方图。',
      bins: [],
      total: 0,
    }
  }

  const count = Number.isInteger(binCount) && binCount > 0 ? binCount : 10
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const usableFlags = Array.isArray(flags) && flags.length === values.length ? flags : null

  if (minimum === maximum) {
    return {
      available: true,
      bins: [
        {
          min: minimum,
          max: maximum,
          count: values.length,
          correct: usableFlags ? usableFlags.filter(Boolean).length : 0,
          wrong: usableFlags ? usableFlags.filter((flag) => !flag).length : 0,
        },
      ],
      total: values.length,
      min: minimum,
      max: maximum,
      constant: true,
    }
  }

  const width = (maximum - minimum) / count
  const bins = Array.from({ length: count }, (_, index) => ({
    min: minimum + index * width,
    max: minimum + (index + 1) * width,
    count: 0,
    correct: 0,
    wrong: 0,
  }))

  values.forEach((value, index) => {
    let binIndex = Math.floor((value - minimum) / width)
    if (binIndex >= count) binIndex = count - 1
    if (binIndex < 0) binIndex = 0
    bins[binIndex].count += 1
    if (usableFlags) {
      if (usableFlags[index]) bins[binIndex].correct += 1
      else bins[binIndex].wrong += 1
    }
  })

  return {
    available: true,
    bins,
    total: values.length,
    min: minimum,
    max: maximum,
    constant: false,
  }
}

/** Confidence/margin histogram with correct-vs-wrong split per bin. */
export function confidenceHistogram(yTrue, yPred, yScore, scoreType, binCount = 10) {
  const confidence = predictionConfidence(yTrue, yPred, yScore, scoreType)
  if (!confidence.available) {
    return {
      available: false,
      reason: confidence.reason,
      title: CONFIDENCE_TITLES.probability.title,
      description: CONFIDENCE_TITLES.probability.description,
      bins: [],
      total: 0,
    }
  }

  const histogram = buildBins(confidence.values, binCount, confidence.correct)
  return {
    available: histogram.available,
    reason: histogram.reason ?? '',
    kind: confidence.kind,
    title: confidence.title,
    description: confidence.description,
    bins: histogram.bins,
    total: histogram.total,
    correctCount: confidence.correctCount,
    wrongCount: confidence.wrongCount,
  }
}

/**
 * Regression diagnostics.
 *
 * ``residual = y_true - y_pred`` (positive means the model under-predicted),
 * ``absolute_error = |residual|``. Sample numbers are 1-based positions inside
 * the test split (``index + 1``), never original dataset row numbers.
 */
export function regressionDiagnostics(yTrue, yPred) {
  if (!Array.isArray(yTrue) || !Array.isArray(yPred)) {
    return unavailable('缺少真实值或预测值数据。')
  }
  if (yTrue.length === 0 || yPred.length === 0) {
    return unavailable('该实验未保存真实值或预测值。')
  }
  if (yTrue.length !== yPred.length) {
    return unavailable('真实值与预测值的样本数量不一致。')
  }
  if (!yTrue.every(isFiniteNumber) || !yPred.every(isFiniteNumber)) {
    return unavailable('真实值或预测值包含非有限数。')
  }

  const actualPredictedPoints = []
  const residualPoints = []
  const residuals = []
  const absoluteErrors = []
  const records = []

  yTrue.forEach((actual, index) => {
    const predicted = yPred[index]
    const residual = actual - predicted
    const absoluteError = Math.abs(residual)

    actualPredictedPoints.push({ index, sample: index + 1, actual, predicted })
    residualPoints.push({ index, sample: index + 1, actual, predicted, residual })
    residuals.push(residual)
    absoluteErrors.push(absoluteError)
    records.push({ index, sample: index + 1, actual, predicted, residual, absoluteError })
  })

  const sortedErrors = [...absoluteErrors].sort((left, right) => left - right)
  const middle = Math.floor(sortedErrors.length / 2)
  const medianAbsoluteError =
    sortedErrors.length % 2 === 0
      ? (sortedErrors[middle - 1] + sortedErrors[middle]) / 2
      : sortedErrors[middle]

  const topAbsoluteErrors = [...records]
    .sort(
      (left, right) =>
        right.absoluteError - left.absoluteError || left.index - right.index,
    )
    .slice(0, 10)

  const meanResidual = residuals.reduce((sum, value) => sum + value, 0) / residuals.length
  const maxAbsoluteError = Math.max(...absoluteErrors)

  return {
    available: true,
    actual_predicted_points: actualPredictedPoints,
    residual_points: residualPoints,
    residuals,
    absolute_errors: absoluteErrors,
    top_absolute_errors: topAbsoluteErrors,
    mean_residual: meanResidual,
    median_absolute_error: medianAbsoluteError,
    max_absolute_error: maxAbsoluteError,
    target_min: Math.min(...yTrue),
    target_max: Math.max(...yTrue),
    sample_count: yTrue.length,
  }
}

/**
 * Objective regression statements built only from real diagnostics.
 *
 * "Near zero" is judged relative to the observed target range (1%), never a
 * hard-coded absolute constant.
 */
export function regressionInsight(diagnostics) {
  if (!diagnostics?.available) {
    return { available: false, statements: [] }
  }

  const statements = []
  const range = diagnostics.target_max - diagnostics.target_min
  const commonlyNearZero =
    range > 0 ? Math.abs(diagnostics.mean_residual) <= range * 0.01 : diagnostics.mean_residual === 0

  if (commonlyNearZero) {
    statements.push(`平均残差为 ${formatValue(diagnostics.mean_residual)}，接近 0。`)
  } else if (diagnostics.mean_residual > 0) {
    statements.push(
      `平均残差为 ${formatValue(diagnostics.mean_residual)}，模型整体存在低估倾向（真实值大于预测值）。`,
    )
  } else {
    statements.push(
      `平均残差为 ${formatValue(diagnostics.mean_residual)}，模型整体存在高估倾向（真实值小于预测值）。`,
    )
  }

  statements.push(
    `中位绝对误差为 ${formatValue(diagnostics.median_absolute_error)}，最大绝对误差为 ${formatValue(
      diagnostics.max_absolute_error,
    )}。`,
  )

  const worst = diagnostics.top_absolute_errors?.[0]
  if (worst) {
    statements.push(
      `最大绝对误差出现在测试集样本 #${worst.sample}（真实值 ${formatValue(
        worst.actual,
      )}，预测值 ${formatValue(worst.predicted)}，残差 ${formatValue(worst.residual)}）。`,
    )
  }

  return { available: true, statements }
}
