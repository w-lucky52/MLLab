/**
 * Phase 6B — training-process helpers contract tests (Node built-in runner).
 *
 * Covers the pure helpers behind the training tab: split validation, curve
 * summary statistics, per-algorithm training flows and no-curve
 * explanations. Run with:
 *   node --test tests/frontend/training_process.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  MISSING_CURVE_HINT,
  curveStatements,
  getAlgorithmTrainingFlow,
  getNoCurveExplanation,
  normalizeTrainingSplit,
  summarizeTrainingCurve,
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

// --- normalizeTrainingSplit ---------------------------------------------------

test('normalizeTrainingSplit computes counts and percentages from real data', () => {
  const split = normalizeTrainingSplit(120, 30)

  assert.equal(split.available, true)
  assert.equal(split.trainCount, 120)
  assert.equal(split.testCount, 30)
  assert.equal(split.total, 150)
  assert.equal(split.trainPercent, 80)
  assert.equal(split.testPercent, 20)
})

test('normalizeTrainingSplit rejects missing, negative, non-integer and non-finite values', () => {
  const invalidInputs = [
    [null, 30],
    [120, undefined],
    [-1, 30],
    [120, -5],
    [1.5, 30],
    [120, 0.5],
    [Number.NaN, 30],
    [Number.POSITIVE_INFINITY, 30],
    [120, Number.NEGATIVE_INFINITY],
    [0, 0],
  ]

  for (const [train, test] of invalidInputs) {
    assert.equal(normalizeTrainingSplit(train, test).available, false, `${train}/${test}`)
  }
})

// --- summarizeTrainingCurve ---------------------------------------------------

test('summarizeTrainingCurve summarizes a normal decreasing curve', () => {
  const summary = summarizeTrainingCurve([10, 6, 4, 3])

  assert.equal(summary.available, true)
  assert.equal(summary.count, 4)
  assert.equal(summary.initial, 10)
  assert.equal(summary.final, 3)
  assert.equal(summary.minimum, 3)
  assert.equal(summary.minRound, 4)
  assert.equal(summary.absoluteChange, -7)
  assert.equal(summary.relativeChangePercent, -70)
  assert.equal(summary.finalVsMinimum, 0)
})

test('summarizeTrainingCurve handles non-monotonic curves', () => {
  const summary = summarizeTrainingCurve([5, 2, 8, 3])

  assert.equal(summary.minimum, 2)
  assert.equal(summary.minRound, 2)
  assert.equal(summary.final, 3)
  assert.equal(summary.absoluteChange, -2)
  assert.equal(summary.finalVsMinimum, 1)
})

test('summarizeTrainingCurve handles constant curves', () => {
  const summary = summarizeTrainingCurve([4, 4, 4])

  assert.equal(summary.available, true)
  assert.equal(summary.absoluteChange, 0)
  assert.equal(summary.relativeChangePercent, 0)
  assert.equal(summary.finalVsMinimum, 0)
  assert.equal(summary.minRound, 1)
})

test('summarizeTrainingCurve handles single-point curves', () => {
  const summary = summarizeTrainingCurve([7])

  assert.equal(summary.available, true)
  assert.equal(summary.count, 1)
  assert.equal(summary.initial, 7)
  assert.equal(summary.final, 7)
  assert.equal(summary.minimum, 7)
  assert.equal(summary.minRound, 1)
  assert.equal(summary.absoluteChange, 0)
  assert.equal(summary.relativeChangePercent, 0)
})

test('summarizeTrainingCurve never divides by a zero initial value', () => {
  const summary = summarizeTrainingCurve([0, 1])

  assert.equal(summary.available, true)
  assert.equal(summary.relativeChangePercent, null)

  const statements = curveStatements(summary)
  assert.ok(statements.some((statement) => statement.includes('相对变化不适用')))
})

test('summarizeTrainingCurve resolves repeated minimums to the first round', () => {
  const summary = summarizeTrainingCurve([5, 3, 3, 4])

  assert.equal(summary.minimum, 3)
  assert.equal(summary.minRound, 2)
})

test('summarizeTrainingCurve rejects empty, nested and non-finite inputs', () => {
  const invalidInputs = [
    null,
    undefined,
    '1,2,3',
    [],
    [
      [1, 2],
      [3, 4],
    ],
    [1, Number.NaN],
    [1, Number.POSITIVE_INFINITY],
    [1, '2'],
  ]

  for (const input of invalidInputs) {
    assert.equal(summarizeTrainingCurve(input).available, false, String(input))
  }
})

test('summarizeTrainingCurve does not modify the input array', () => {
  const input = [3, 1, 2, 1]
  const snapshot = [...input]
  summarizeTrainingCurve(input)

  assert.deepEqual(input, snapshot)
})

// --- curveStatements ----------------------------------------------------------

test('curveStatements describes direction objectively', () => {
  const decreasing = curveStatements(summarizeTrainingCurve([10, 3]))
  assert.ok(decreasing.some((statement) => statement.includes('最终值低于初始值')))

  const increasing = curveStatements(summarizeTrainingCurve([1, 5]))
  assert.ok(increasing.some((statement) => statement.includes('最终值高于初始值')))

  const constant = curveStatements(summarizeTrainingCurve([4, 4]))
  assert.ok(constant.some((statement) => statement.includes('最终值等于初始值')))

  const minimum = curveStatements(summarizeTrainingCurve([5, 2, 8]))
  assert.ok(minimum.some((statement) => statement.includes('最低记录出现在第 2 轮')))
})

test('curveStatements returns nothing for unavailable summaries', () => {
  assert.deepEqual(curveStatements(summarizeTrainingCurve([])), [])
  assert.deepEqual(curveStatements(null), [])
})

// --- getAlgorithmTrainingFlow -------------------------------------------------

test('KNN flow explains neighbor search and reads recorded parameters', () => {
  const flow = getAlgorithmTrainingFlow('knn_classifier', null)

  assert.equal(flow.available, true)
  assert.deepEqual(flow.steps, [
    '保存训练样本',
    '计算待预测样本与训练样本的距离',
    '选择最近的 K 个邻居',
    '按 weights 规则投票',
    '输出类别',
  ])
  assert.ok(flow.notes.some((note) => note.includes('没有基于梯度的参数优化过程')))
  assert.ok(flow.notes.some((note) => note.includes('没有逐轮损失曲线')))
  assert.ok(flow.notes.some((note) => note.includes('训练阶段主要保存训练数据')))

  // Missing parameters must never be filled with registry defaults.
  const joined = flow.notes.join('\n')
  assert.equal((joined.match(/未记录/g) || []).length, 3)

  const recorded = getAlgorithmTrainingFlow('knn_classifier', {
    n_neighbors: 5,
    weights: 'distance',
    p: 2,
  })
  const recordedJoined = recorded.notes.join('\n')
  assert.ok(recordedJoined.includes('n_neighbors = 5'))
  assert.ok(recordedJoined.includes('weights = distance'))
  assert.ok(recordedJoined.includes('p = 2'))
  assert.ok(!recordedJoined.includes('未记录'))
})

test('GaussianNB flow explains statistical estimation', () => {
  const flow = getAlgorithmTrainingFlow('gaussian_nb', null)

  assert.deepEqual(flow.steps, [
    '按类别划分训练样本',
    '估计每个类别的特征均值与方差',
    '估计类别先验',
    '计算后验相关分数',
    '输出概率最大的类别',
  ])
  assert.ok(flow.notes.some((note) => note.includes('统计估计')))
  assert.ok(flow.notes.some((note) => note.includes('不执行梯度迭代')))
  assert.ok(flow.notes.some((note) => note.includes('var_smoothing = 未记录')))

  const recorded = getAlgorithmTrainingFlow('gaussian_nb', { var_smoothing: 1e-9 })
  assert.ok(recorded.notes.some((note) => note.includes('var_smoothing = 1e-9')))
})

test('decision tree flow explains recursive splitting without fake rounds', () => {
  const flow = getAlgorithmTrainingFlow('decision_tree_regressor', null)

  assert.deepEqual(flow.steps, [
    '搜索候选特征与划分阈值',
    '选择使节点误差下降的划分',
    '递归生成子节点',
    '满足停止条件后生成叶节点',
    '叶节点输出预测值',
  ])
  assert.ok(flow.notes.some((note) => note.includes('递归划分')))
  assert.ok(flow.notes.some((note) => note.includes('不代表迭代轮数')))
})

test('random forest flow explains tree aggregation without fake iterations', () => {
  const flow = getAlgorithmTrainingFlow('random_forest_regressor', { n_estimators: 120 })

  assert.deepEqual(flow.steps, [
    '生成多组训练子样本',
    '为每组样本训练一棵决策树',
    '各棵树独立产生预测',
    '聚合多棵树的预测结果',
    '输出平均预测值',
  ])
  assert.ok(flow.notes.some((note) => note.includes('多棵决策树')))
  assert.ok(flow.notes.some((note) => note.includes('n_estimators = 120')))
  assert.ok(flow.notes.some((note) => note.includes('不是优化迭代次数')))
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
    const flow = getAlgorithmTrainingFlow(modelName, null)
    assert.equal(flow.available, true)
    assert.equal(flow.steps.length, 5, modelName)
    assert.ok(flow.steps.includes(keySteps[modelName]), modelName)
  }
})

test('GBDT flow states the curve is a training loss, not test error', () => {
  const flow = getAlgorithmTrainingFlow('gbdt_regressor', null)

  assert.ok(flow.notes.some((note) => note.includes('train_score_')))
  assert.ok(flow.notes.some((note) => note.includes('训练集损失')))
  assert.ok(flow.notes.some((note) => note.includes('不代表测试集误差')))
})

test('unknown models fall back to a safe generic flow', () => {
  const flow = getAlgorithmTrainingFlow('some_unknown_model', { foo: 1 })

  assert.equal(flow.available, true)
  assert.ok(flow.steps.length > 0)
  assert.ok(flow.notes.length > 0)

  assert.deepEqual(getNoCurveExplanation('some_unknown_model'), ['该模型未提供逐轮损失曲线。'])
})

// --- getNoCurveExplanation ------------------------------------------------------

test('no-curve explanations exist for every non-iterative model', () => {
  const keywords = {
    knn_classifier: '基于梯度',
    gaussian_nb: '统计估计',
    decision_tree_regressor: '递归划分',
    random_forest_regressor: '多棵决策树',
  }

  for (const modelName of NON_ITERATIVE_MODELS) {
    const statements = getNoCurveExplanation(modelName)
    assert.ok(statements.length >= 2, modelName)
    assert.ok(
      statements.some((statement) => statement.includes(keywords[modelName])),
      modelName,
    )
    assert.ok(
      statements.some((statement) => statement.includes('损失曲线')),
      modelName,
    )
  }
})

// --- global wording and data-integrity guards -----------------------------------

test('no flow or explanation uses subjective quality wording', () => {
  const texts = []

  for (const modelName of [...ITERATIVE_MODELS, ...NON_ITERATIVE_MODELS, 'unknown']) {
    const flow = getAlgorithmTrainingFlow(modelName, null)
    texts.push(flow.title, ...flow.steps, ...flow.notes)
    texts.push(...getNoCurveExplanation(modelName))
  }

  texts.push(...curveStatements(summarizeTrainingCurve([3, 1, 2])))

  assert.doesNotMatch(texts.join('\n'), BANNED_WORDS)
})

test('non-iterative flows never generate simulated loss data', () => {
  for (const modelName of NON_ITERATIVE_MODELS) {
    const flow = getAlgorithmTrainingFlow(modelName, null)

    assert.deepEqual(Object.keys(flow).sort(), ['available', 'notes', 'steps', 'title'])
    for (const value of Object.values(flow)) {
      if (Array.isArray(value)) {
        assert.ok(
          value.every((item) => typeof item === 'string'),
          `${modelName} must only carry string lists`,
        )
      }
    }
  }
})

test('missing-curve hint matches the agreed wording', () => {
  assert.equal(MISSING_CURVE_HINT, '该历史实验未保存训练曲线，无法展示逐轮变化。')
})
