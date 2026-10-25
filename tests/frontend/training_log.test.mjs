/**
 * Phase 7B1 — training-process log display helpers (Node built-in runner).
 *
 * Covers the Chinese line generation from structured entries: every phase
 * label, per-event messages, elapsed formatting, safe fallbacks and the
 * no-fabrication rules (iteration checkpoints never show a fake duration,
 * invalid input hides the panel, inputs are never mutated).
 *
 * Run with:
 *   node --test tests/frontend/training_log.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  lossNameLabel,
  metricLabel,
  normalizeProcessLog,
  phaseLabel,
  processLogLines,
  processLogText,
} from '../../frontend/src/utils/resultLabels.js'

// --- phase / loss / metric labels ---------------------------------------------

test('every known event code has a Chinese phase label', () => {
  const expected = {
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
  for (const [event, label] of Object.entries(expected)) {
    assert.equal(phaseLabel(event), label, `missing label for ${event}`)
  }
  assert.equal(phaseLabel('something_unknown'), '系统')
})

test('loss name and metric labels map with safe fallbacks', () => {
  assert.equal(lossNameLabel('objective'), '目标函数')
  assert.equal(lossNameLabel('training_loss'), '训练损失')
  assert.equal(lossNameLabel('something_else'), '损失值')
  assert.equal(lossNameLabel(undefined), '损失值')

  assert.equal(metricLabel('accuracy'), 'Accuracy')
  assert.equal(metricLabel('precision_weighted'), '精确率')
  assert.equal(metricLabel('recall_weighted'), '召回率')
  assert.equal(metricLabel('f1_weighted'), 'F1')
  assert.equal(metricLabel('mae'), 'MAE')
  assert.equal(metricLabel('mse'), 'MSE')
  assert.equal(metricLabel('rmse'), 'RMSE')
  assert.equal(metricLabel('r2'), 'R²')
  assert.equal(metricLabel('custom_metric'), 'custom_metric')
})

// --- per-event Chinese lines ----------------------------------------------------

test('dataset_loaded renders real dataset shape', () => {
  const [line] = processLogLines([
    {
      sequence: 1,
      event: 'dataset_loaded',
      elapsed_ms: 2.41,
      details: {
        dataset_name: 'iris',
        sample_count: 150,
        feature_count: 4,
        task_type: 'classification',
      },
    },
  ])
  assert.equal(line.phase, '数据')
  assert.equal(line.message, '加载数据集 iris：150 个样本，4 个特征')
})

test('data_prepared renders split, stratification and standardization', () => {
  const [line] = processLogLines([
    {
      sequence: 2,
      event: 'data_prepared',
      elapsed_ms: 5,
      details: {
        train_sample_count: 105,
        test_sample_count: 45,
        test_size: 0.3,
        random_state: 42,
        standardize: true,
        stratified: true,
      },
    },
  ])
  assert.equal(line.message, '数据准备完成：训练 105，测试 45，使用分层切分，并完成标准化')
})

test('data_prepared omits stratify/standardize wording when false', () => {
  const [line] = processLogLines([
    {
      sequence: 2,
      event: 'data_prepared',
      elapsed_ms: 5,
      details: { train_sample_count: 309, test_sample_count: 133, standardize: true, stratified: false },
    },
  ])
  assert.equal(line.message, '数据准备完成：训练 309，测试 133，并完成标准化')
})

test('iteration_checkpoint renders the real loss value without fake timing', () => {
  const [line] = processLogLines([
    {
      sequence: 5,
      event: 'iteration_checkpoint',
      elapsed_ms: null,
      details: { iteration: 1, value: 1.050966, loss_name: 'objective', source: 'loss_history_' },
    },
  ])
  assert.equal(line.phase, '迭代')
  assert.equal(line.message, '第 1 轮：目标函数 = 1.05097')
  assert.equal(line.elapsedText, null)
})

test('iteration_checkpoint never shows a duration even if one is present', () => {
  const [line] = processLogLines([
    {
      sequence: 5,
      event: 'iteration_checkpoint',
      elapsed_ms: 12.5,
      details: { iteration: 2, value: 0.5, loss_name: 'training_loss', source: 'train_score_' },
    },
  ])
  assert.equal(line.elapsedText, null)
  assert.equal(line.message, '第 2 轮：训练损失 = 0.5')
})

test('training_completed renders rounds and stop reason', () => {
  const [line] = processLogLines([
    {
      sequence: 8,
      event: 'training_completed',
      elapsed_ms: 120,
      details: { fit_duration_ms: 118, n_iter: 1000, stop_reason: 'max_iter', loss_point_count: 25 },
    },
  ])
  assert.equal(line.message, '训练结束：共 1000 轮，停止原因为达到最大迭代次数')

  const [converged] = processLogLines([
    {
      sequence: 8,
      event: 'training_completed',
      elapsed_ms: 30,
      details: { n_iter: 12, stop_reason: 'converged' },
    },
  ])
  assert.equal(converged.message, '训练结束：共 12 轮，停止原因为已收敛')

  const [noReason] = processLogLines([
    { sequence: 8, event: 'training_completed', elapsed_ms: 30, details: { n_iter: 7, stop_reason: null } },
  ])
  assert.equal(noReason.message, '训练结束：共 7 轮')
  assert.ok(!noReason.message.includes('null'))
})

test('classification evaluation renders metric abbreviations and Chinese labels', () => {
  const [line] = processLogLines([
    {
      sequence: 10,
      event: 'evaluation_completed',
      elapsed_ms: 150,
      details: { evaluation_duration_ms: 12, metrics: { accuracy: 0.955556, f1_weighted: 0.955357 } },
    },
  ])
  assert.equal(line.message, 'Accuracy 0.955556，F1 0.955357')
})

test('regression evaluation renders regression metric labels', () => {
  const [line] = processLogLines([
    {
      sequence: 10,
      event: 'evaluation_completed',
      elapsed_ms: 150,
      details: { metrics: { mae: 42.5, rmse: 55.1234567, r2: 0.45 } },
    },
  ])
  assert.equal(line.message, 'MAE 42.5，RMSE 55.1235，R² 0.45')
})

test('prediction and completion events render counts', () => {
  const lines = processLogLines([
    { sequence: 9, event: 'prediction_completed', elapsed_ms: 140, details: { prediction_count: 45 } },
    { sequence: 11, event: 'experiment_completed', elapsed_ms: 200, details: { chart_count: 5, total_duration_ms: 200 } },
  ])
  assert.equal(lines[0].message, '完成 45 个测试样本预测')
  assert.equal(lines[1].message, '实验处理完成')
})

test('experiment_completed never reports a backend chart count', () => {
  // Charts on the result page are frontend ECharts, so any backend
  // chart_count (including legacy 0) must be ignored in the display text.
  for (const chartCount of [0, 3, 5, 12]) {
    const [line] = processLogLines([
      {
        sequence: 1,
        event: 'experiment_completed',
        elapsed_ms: 10,
        details: { chart_count: chartCount, total_duration_ms: 10 },
      },
    ])
    assert.equal(line.message, '实验处理完成')
    assert.ok(!line.message.includes('张图表'), line.message)
    assert.ok(!line.message.includes('共生成'), line.message)
    assert.ok(!line.message.includes('生成 0 张图表'), line.message)
  }

  const [withoutCount] = processLogLines([
    { sequence: 1, event: 'experiment_completed', elapsed_ms: 10, details: { total_duration_ms: 10 } },
  ])
  assert.equal(withoutCount.message, '实验处理完成')
})

// --- elapsed formatting ----------------------------------------------------------

test('elapsed_ms is formatted as a human readable duration', () => {
  const [line] = processLogLines([
    { sequence: 1, event: 'dataset_loaded', elapsed_ms: 2.41, details: {} },
  ])
  assert.equal(line.elapsedText, '2.41 ms')

  const [slow] = processLogLines([
    { sequence: 1, event: 'training_completed', elapsed_ms: 1500, details: {} },
  ])
  assert.equal(slow.elapsedText, '1.5 s')
})

// --- defensive behaviour -----------------------------------------------------------

test('unknown events fall back to 系统 with the raw event as message', () => {
  const [line] = processLogLines([
    { sequence: 1, event: 'mystery_event', elapsed_ms: 1, details: {} },
  ])
  assert.equal(line.phase, '系统')
  assert.equal(line.message, 'mystery_event')
})

test('missing details never produce undefined text', () => {
  const events = [
    'dataset_loaded',
    'data_prepared',
    'estimator_created',
    'training_started',
    'iteration_checkpoint',
    'training_completed',
    'prediction_completed',
    'evaluation_completed',
    'experiment_completed',
    'experiment_failed',
  ]
  const lines = processLogLines(events.map((event, index) => ({ sequence: index + 1, event })))
  assert.equal(lines.length, events.length)
  for (const line of lines) {
    assert.ok(!line.message.includes('undefined'), line.message)
    assert.ok(!line.message.includes('null'), line.message)
    assert.ok(!line.message.includes('NaN'), line.message)
  }
})

test('non-finite values never reach the message', () => {
  const lines = processLogLines([
    {
      sequence: 1,
      event: 'evaluation_completed',
      elapsed_ms: 1,
      details: { metrics: { accuracy: Number.NaN, f1_weighted: Number.POSITIVE_INFINITY } },
    },
    {
      sequence: 2,
      event: 'iteration_checkpoint',
      elapsed_ms: null,
      details: { iteration: 1, value: Number.NaN },
    },
  ])
  for (const line of lines) {
    assert.ok(!line.message.includes('NaN'), line.message)
    assert.ok(!line.message.includes('Infinity'), line.message)
  }
  assert.equal(lines[0].message, '评估完成')
})

test('normalisation sorts by sequence and drops invalid entries', () => {
  const lines = processLogLines([
    { sequence: 3, event: 'training_started', elapsed_ms: 9, details: {} },
    { sequence: 1, event: 'dataset_loaded', elapsed_ms: 1, details: {} },
    { sequence: -2, event: 'bad_negative', elapsed_ms: 1, details: {} },
    { sequence: 2, event: 'data_prepared', elapsed_ms: Number.NaN, details: {} },
    { sequence: 4, event: '', elapsed_ms: 1, details: {} },
    'not-an-entry',
  ])
  assert.deepEqual(
    lines.map((line) => line.sequence),
    [1, 3],
  )
  assert.equal(lines[0].event, undefined) // display lines carry phase/message, not event
  assert.equal(lines[0].phase, '数据')
  assert.equal(lines[1].phase, '训练')
})

test('the input log is never mutated', () => {
  const raw = [
    {
      sequence: 2,
      event: 'training_started',
      elapsed_ms: 5,
      details: { model_name: 'mlp_classifier', train_sample_count: 105 },
    },
    { sequence: 1, event: 'dataset_loaded', elapsed_ms: 1, details: { dataset_name: 'iris' } },
  ]
  const snapshot = JSON.parse(JSON.stringify(raw))
  processLogLines(raw)
  assert.deepEqual(raw, snapshot)
})

test('invalid or empty process_log yields no lines so the panel hides', () => {
  for (const raw of [null, undefined, {}, 'x', 5, [], [null], [{ sequence: 0, event: 'x' }]]) {
    assert.deepEqual(processLogLines(raw), [], `expected empty lines for ${JSON.stringify(raw)}`)
  }
})

test('processLogText mirrors the displayed lines for the copy button', () => {
  const lines = processLogLines([
    {
      sequence: 1,
      event: 'dataset_loaded',
      elapsed_ms: 2.41,
      details: { dataset_name: 'iris', sample_count: 150, feature_count: 4 },
    },
    {
      sequence: 2,
      event: 'iteration_checkpoint',
      elapsed_ms: null,
      details: { iteration: 1, value: 1.050966, loss_name: 'objective' },
    },
  ])
  const text = processLogText(lines)
  assert.equal(
    text,
    '#1 [数据] 加载数据集 iris：150 个样本，4 个特征（2.41 ms）\n' +
      '#2 [迭代] 第 1 轮：目标函数 = 1.05097',
  )
  assert.equal(processLogText([]), '')
  assert.equal(processLogText(null), '')
})

test('normalizeProcessLog keeps the contract fields only', () => {
  const [entry] = normalizeProcessLog([
    { sequence: 1, event: 'dataset_loaded', elapsed_ms: 1, details: {}, extra: 'dropped' },
  ])
  assert.deepEqual(Object.keys(entry).sort(), ['details', 'elapsed_ms', 'event', 'sequence'])
})
