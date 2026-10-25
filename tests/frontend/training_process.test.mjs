/**
 * Phase 7A — training-process helpers contract tests (Node built-in runner).
 *
 * Covers the pure helpers behind the training tab after the subtraction
 * refactor: summary cards (duration / split / scale / standardize), loss-curve
 * normalization and statistics for the chart header, and the per-algorithm
 * step flows. Run with:
 *   node --test tests/frontend/training_process.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  UNRECORDED,
  curveSummary,
  formatDuration,
  formatNumber,
  formatSampleSplit,
  getAlgorithmTrainingFlow,
  isIterativeModel,
  iterationSummary,
  modelScaleSummary,
  normalizeCurve,
  standardizeSummary,
} from '../../frontend/src/utils/training.js'

const BANNED_WORDS = /很好|很差|过拟合|泛化良好|最优模型/

const ITERATIVE_MODELS = [
  'linear_regression',
  'ridge_regression',
  'logistic_regression',
  'linear_svm',
  'mlp_classifier',
  'gbdt_regressor',
]

const NON_ITERATIVE_MODELS = [
  'knn_classifier',
  'gaussian_nb',
  'decision_tree_regressor',
  'random_forest_regressor',
]

// --- formatting helpers -------------------------------------------------------

test('formatDuration renders milliseconds and seconds', () => {
  assert.equal(formatDuration(320), '320 ms')
  assert.equal(formatDuration(1520), '1.52 s')
  assert.equal(formatDuration(null), UNRECORDED)
  assert.equal(formatDuration(-5), UNRECORDED)
  assert.equal(formatDuration('abc'), UNRECORDED)
})

test('formatSampleSplit joins recorded train and test counts', () => {
  assert.equal(formatSampleSplit({ train_sample_count: 120, test_sample_count: 30 }), '训练 120 / 测试 30')
  assert.equal(formatSampleSplit({ train_sample_count: 120 }), '训练 120')
  assert.equal(formatSampleSplit({}), UNRECORDED)
  assert.equal(formatSampleSplit(null), UNRECORDED)
})

test('formatNumber never leaks NaN or undefined text', () => {
  assert.equal(formatNumber(null), UNRECORDED)
  assert.equal(formatNumber(undefined), UNRECORDED)
  assert.equal(formatNumber(Number.NaN), UNRECORDED)
  assert.equal(formatNumber(Number.POSITIVE_INFINITY), UNRECORDED)
  assert.equal(formatNumber(''), UNRECORDED)
  assert.equal(formatNumber(4), '4')
  assert.equal(formatNumber(0.123456789), '0.123457')
})

// --- iterationSummary -----------------------------------------------------------

test('iterationSummary prefers recorded n_iter over anything else', () => {
  assert.equal(
    iterationSummary({ n_iter: 57 }, { max_iter: 200 }),
    '57 / 200',
  )
  assert.equal(iterationSummary({ n_iter: 57 }, null), '57 次')
})

test('iterationSummary falls back to recorded train_score_ rounds', () => {
  const info = { loss_source: 'train_score_', loss_history: [1, 0.9, 0.8] }
  assert.equal(iterationSummary(info, null), '记录 3 轮')
})

test('iterationSummary never invents rounds for missing data', () => {
  assert.equal(iterationSummary({}, null), '不适用')
  assert.equal(iterationSummary({ loss_source: 'train_score_', loss_history: [] }, null), '不适用')
})

// --- standardizeSummary ---------------------------------------------------------

test('standardizeSummary maps recorded booleans and stays unrecorded otherwise', () => {
  assert.equal(standardizeSummary(true), '已开启')
  assert.equal(standardizeSummary(false), '未开启')
  assert.equal(standardizeSummary(undefined), UNRECORDED)
  assert.equal(standardizeSummary(null), UNRECORDED)
})

// --- modelScaleSummary ------------------------------------------------------------

test('modelScaleSummary shows iterations for iterative models', () => {
  for (const modelName of ITERATIVE_MODELS) {
    const result = modelScaleSummary(modelName, { n_iter: 12 }, { max_iter: 100 }, null)
    assert.equal(result.label, '迭代情况', modelName)
    assert.equal(result.value, '12 / 100', modelName)
  }
})

test('modelScaleSummary reads K for KNN from recorded parameters only', () => {
  const resolved = modelScaleSummary('knn_classifier', {}, { n_neighbors: 7 }, null)
  assert.equal(resolved.label, '邻居数量 K')
  assert.equal(resolved.value, '7')

  const fallback = modelScaleSummary('knn_classifier', {}, null, { n_neighbors: 3 })
  assert.equal(fallback.value, '3')

  const missing = modelScaleSummary('knn_classifier', {}, null, null)
  assert.equal(missing.value, UNRECORDED)
})

test('modelScaleSummary shows the recorded class count for GaussianNB', () => {
  const result = modelScaleSummary('gaussian_nb', { classes: ['a', 'b', 'c'] }, null, null)
  assert.equal(result.label, '类别数')
  assert.equal(result.value, '3')

  assert.equal(modelScaleSummary('gaussian_nb', {}, null, null).value, UNRECORDED)
})

test('modelScaleSummary shows max_depth for decision trees without backfilling defaults', () => {
  const result = modelScaleSummary('decision_tree_regressor', {}, { max_depth: 6 }, null)
  assert.equal(result.label, '最大深度')
  assert.equal(result.value, '6')

  const fallback = modelScaleSummary('decision_tree_regressor', {}, null, { max_depth: 4 })
  assert.equal(fallback.value, '4')

  // A recorded null (unlimited depth) is not a number: show 未记录, never a default.
  const unlimited = modelScaleSummary('decision_tree_regressor', {}, { max_depth: null }, null)
  assert.equal(unlimited.value, UNRECORDED)
})

test('modelScaleSummary shows tree count for random forests', () => {
  const result = modelScaleSummary('random_forest_regressor', {}, { n_estimators: 120 }, null)
  assert.equal(result.label, '树数量')
  assert.equal(result.value, '120')

  assert.equal(modelScaleSummary('random_forest_regressor', {}, null, null).value, UNRECORDED)
})

test('modelScaleSummary stays unrecorded for unknown models', () => {
  const result = modelScaleSummary('some_unknown_model', {}, null, null)
  assert.equal(result.label, '模型规模')
  assert.equal(result.value, UNRECORDED)
})

// --- normalizeCurve / curveSummary ----------------------------------------------

test('normalizeCurve returns a finite copy and rejects bad input', () => {
  const input = [3, 1, 2]
  const curve = normalizeCurve(input)
  assert.deepEqual(curve, [3, 1, 2])
  assert.notEqual(curve, input)

  for (const bad of [null, undefined, [], '1,2', [1, Number.NaN], [1, '2'], [[1]]]) {
    assert.equal(normalizeCurve(bad), null, String(bad))
  }
})

test('curveSummary reports count, initial, final and minimum', () => {
  const summary = curveSummary([10, 6, 4, 3])
  assert.deepEqual(summary, { count: 4, initial: 10, final: 3, minimum: 3 })

  assert.deepEqual(curveSummary([7]), { count: 1, initial: 7, final: 7, minimum: 7 })
  assert.equal(curveSummary([]), null)
  assert.equal(curveSummary(null), null)
})

// --- getAlgorithmTrainingFlow -------------------------------------------------

test('KNN flow explains neighbor search as compact steps', () => {
  const flow = getAlgorithmTrainingFlow('knn_classifier')

  assert.equal(flow.available, true)
  assert.deepEqual(flow.steps, [
    '保存训练样本',
    '计算待预测样本与训练样本的距离',
    '选择最近的 K 个邻居',
    '按 weights 规则投票',
    '输出类别',
  ])
})

test('GaussianNB flow explains statistical estimation as compact steps', () => {
  const flow = getAlgorithmTrainingFlow('gaussian_nb')

  assert.deepEqual(flow.steps, [
    '按类别划分训练样本',
    '估计每个类别的特征均值与方差',
    '估计类别先验',
    '计算后验相关分数',
    '输出概率最大的类别',
  ])
})

test('decision tree flow explains recursive splitting as compact steps', () => {
  const flow = getAlgorithmTrainingFlow('decision_tree_regressor')

  assert.deepEqual(flow.steps, [
    '搜索候选特征与划分阈值',
    '选择使节点误差下降的划分',
    '递归生成子节点',
    '满足停止条件后生成叶节点',
    '叶节点输出预测值',
  ])
})

test('random forest flow explains tree aggregation as compact steps', () => {
  const flow = getAlgorithmTrainingFlow('random_forest_regressor')

  assert.deepEqual(flow.steps, [
    '生成多组训练子样本',
    '为每组样本训练一棵决策树',
    '各棵树独立产生预测',
    '聚合多棵树的预测结果',
    '输出平均预测值',
  ])
})

test('iterative models expose their gradient-based training mechanisms', () => {
  const keySteps = {
    linear_regression: '计算均方目标梯度',
    ridge_regression: '加入 L2 正则项',
    logistic_regression: '计算 Softmax 概率',
    linear_svm: '计算 hinge 次梯度',
    mlp_classifier: '反向传播',
    gbdt_regressor: '拟合新的弱学习器',
  }

  for (const modelName of ITERATIVE_MODELS) {
    const flow = getAlgorithmTrainingFlow(modelName)
    assert.equal(flow.available, true)
    assert.equal(flow.steps.length, 5, modelName)
    assert.ok(flow.steps.includes(keySteps[modelName]), modelName)
  }
})

test('unknown models fall back to a safe generic flow', () => {
  const flow = getAlgorithmTrainingFlow('some_unknown_model')

  assert.equal(flow.available, true)
  assert.ok(flow.steps.length > 0)
})

// --- global wording and data-integrity guards -----------------------------------

test('no flow uses subjective quality wording', () => {
  const texts = []

  for (const modelName of [...ITERATIVE_MODELS, ...NON_ITERATIVE_MODELS, 'unknown']) {
    const flow = getAlgorithmTrainingFlow(modelName)
    texts.push(flow.title, ...flow.steps)
  }

  assert.doesNotMatch(texts.join('\n'), BANNED_WORDS)
})

test('flows never generate simulated loss data', () => {
  for (const modelName of [...ITERATIVE_MODELS, ...NON_ITERATIVE_MODELS, 'unknown']) {
    const flow = getAlgorithmTrainingFlow(modelName)

    assert.deepEqual(Object.keys(flow).sort(), ['available', 'steps', 'title'])
    assert.ok(flow.steps.every((item) => typeof item === 'string'), modelName)
  }
})

test('isIterativeModel matches the agreed model split', () => {
  for (const modelName of ITERATIVE_MODELS) {
    assert.equal(isIterativeModel(modelName), true, modelName)
  }
  for (const modelName of NON_ITERATIVE_MODELS) {
    assert.equal(isIterativeModel(modelName), false, modelName)
  }
  assert.equal(isIterativeModel('some_unknown_model'), false)
})
