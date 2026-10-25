/**
 * Pure helpers for the result page: structured training-process log
 * normalisation and Chinese display labels.
 *
 * The backend stores structured entries (``event`` + ``details``), never
 * pre-rendered sentences — every Chinese line below is generated here from
 * the structured payload. All functions are defensive: invalid input yields
 * an empty array / safe fallback, inputs are never mutated, and no NaN /
 * Infinity / undefined text can reach the page.
 */

import { formatDuration, formatNumber, toFiniteNumber, toInteger } from './training.js'

/** Phase label per event code; unknown events fall back to 系统. */
const PHASE_LABELS = {
  dataset_loaded: '数据',
  data_prepared: '预处理',
  estimator_created: '模型',
  training_started: '训练',
  iteration_checkpoint: '迭代',
  training_completed: '完成',
  prediction_completed: '预测',
  evaluation_completed: '评估',
  experiment_completed: '结束',
  experiment_failed: '失败',
}

export function phaseLabel(event) {
  return PHASE_LABELS[event] || '系统'
}

/** Chinese label for the recorded loss curve name. */
export function lossNameLabel(lossName) {
  if (lossName === 'objective') return '目标函数'
  if (lossName === 'training_loss') return '训练损失'
  return '损失值'
}

/** Display label for a scalar metric key (common abbreviations stay). */
export function metricLabel(key) {
  const labels = {
    accuracy: 'Accuracy',
    precision_weighted: '精确率',
    recall_weighted: '召回率',
    f1_weighted: 'F1',
    mae: 'MAE',
    mse: 'MSE',
    rmse: 'RMSE',
    r2: 'R²',
  }
  return labels[key] || String(key)
}

const STOP_REASON_LABELS = {
  converged: '已收敛',
  max_iter: '达到最大迭代次数',
}

const isPlainObject = (value) =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

/**
 * Validate and normalise a raw ``process_log`` payload.
 *
 * Returns a new array sorted by ``sequence``; entries keep only the four
 * contract fields with safe fallbacks. Returns ``[]`` unless the input is an
 * array with at least one valid entry. The input is never mutated.
 */
export function normalizeProcessLog(raw) {
  if (!Array.isArray(raw) || raw.length === 0) return []

  const entries = []
  for (const item of raw) {
    if (!isPlainObject(item)) continue
    const sequence = toInteger(item.sequence)
    if (sequence === null || sequence < 1) continue
    if (typeof item.event !== 'string' || !item.event) continue

    let elapsed = null
    if (item.elapsed_ms !== null && item.elapsed_ms !== undefined) {
      const number = toFiniteNumber(item.elapsed_ms)
      if (number === null || number < 0) continue
      elapsed = number
    }

    entries.push({
      sequence,
      event: item.event,
      elapsed_ms: elapsed,
      details: isPlainObject(item.details) ? { ...item.details } : {},
    })
  }

  entries.sort((a, b) => a.sequence - b.sequence)
  return entries
}

/** ``value`` rendered with ``suffix`` when it is a finite number, else ``''``. */
const withNumber = (value, render) => {
  const number = toFiniteNumber(value)
  return number === null ? '' : render(number)
}

const withInteger = (value, render) => {
  const integer = toInteger(value)
  return integer === null ? '' : render(integer)
}

/** Build the Chinese message for one normalised entry. */
function buildMessage(entry) {
  const details = entry.details
  switch (entry.event) {
    case 'dataset_loaded': {
      const name = typeof details.dataset_name === 'string' ? details.dataset_name : ''
      const samples = withInteger(details.sample_count, (n) => `${n} 个样本`)
      const features = withInteger(details.feature_count, (n) => `${n} 个特征`)
      const counts = [samples, features].filter(Boolean).join('，')
      return `加载数据集 ${name}${counts ? `：${counts}` : ''}`.trim()
    }
    case 'data_prepared': {
      const parts = []
      const train = withInteger(details.train_sample_count, (n) => `训练 ${n}`)
      const test = withInteger(details.test_sample_count, (n) => `测试 ${n}`)
      if (train && test) parts.push(`${train}，${test}`)
      else if (train || test) parts.push(train || test)
      if (details.stratified === true) parts.push('使用分层切分')
      if (details.standardize === true) parts.push('并完成标准化')
      return parts.length ? `数据准备完成：${parts.join('，')}` : '数据准备完成'
    }
    case 'estimator_created': {
      const klass =
        typeof details.estimator_class === 'string' && details.estimator_class
          ? details.estimator_class
          : typeof details.model_name === 'string'
            ? details.model_name
            : ''
      return klass ? `创建 ${klass}` : '创建模型'
    }
    case 'training_started': {
      const count = withInteger(details.train_sample_count, (n) => `使用 ${n} 个样本`)
      const name = typeof details.model_name === 'string' ? details.model_name : ''
      return `开始${count}训练 ${name}`.trim()
    }
    case 'iteration_checkpoint': {
      const iteration = withInteger(details.iteration, (n) => `第 ${n} 轮`)
      const value = withNumber(details.value, (n) => formatNumber(n))
      if (!iteration || !value) return '迭代检查点'
      return `${iteration}：${lossNameLabel(details.loss_name)} = ${value}`
    }
    case 'training_completed': {
      let message = '训练结束'
      const rounds = withInteger(details.n_iter, (n) => `共 ${n} 轮`)
      if (rounds) message += `：${rounds}`
      const reason = STOP_REASON_LABELS[details.stop_reason]
      if (reason) message += `${rounds ? '，' : '：'}停止原因为${reason}`
      return message
    }
    case 'prediction_completed':
      return (
        withInteger(details.prediction_count, (n) => `完成 ${n} 个测试样本预测`) ||
        '预测完成'
      )
    case 'evaluation_completed': {
      if (!isPlainObject(details.metrics)) return '评估完成'
      const parts = []
      for (const [key, value] of Object.entries(details.metrics)) {
        const number = toFiniteNumber(value)
        if (number === null) continue
        parts.push(`${metricLabel(key)} ${formatNumber(number)}`)
      }
      return parts.length ? parts.join('，') : '评估完成'
    }
    case 'experiment_completed':
      // Charts are rendered by the frontend ECharts layer, so the backend
      // chart-file count says nothing about what the result page shows.
      // Never surface a count here (legacy logs may still carry chart_count).
      return '实验处理完成'
    case 'experiment_failed':
      return '实验失败'
    default:
      return entry.event
  }
}

/**
 * Format one normalised entry for display.
 *
 * ``elapsedText`` is ``null`` whenever no real timing exists (always the
 * case for ``iteration_checkpoint``), so no fake duration is ever shown.
 */
export function formatProcessLogEntry(entry) {
  const phase = phaseLabel(entry?.event)
  return {
    sequence: entry.sequence,
    phase,
    elapsedText:
      entry.event === 'iteration_checkpoint' || entry.elapsed_ms === null
        ? null
        : formatDuration(entry.elapsed_ms),
    message: buildMessage(entry),
  }
}

/**
 * Normalise ``raw`` and return display-ready lines
 * (``{ sequence, phase, elapsedText, message }``), or ``[]`` when the log is
 * missing or invalid — the caller then hides the log panel entirely.
 */
export function processLogLines(raw) {
  return normalizeProcessLog(raw).map(formatProcessLogEntry)
}

/** Plain-text rendering of the displayed lines, used by "复制日志". */
export function processLogText(lines) {
  if (!Array.isArray(lines)) return ''
  return lines
    .map((line) => {
      const time = line.elapsedText ? `（${line.elapsedText}）` : ''
      return `#${line.sequence} [${line.phase}] ${line.message}${time}`
    })
    .join('\n')
}
