/**
 * Phase 7A — result page structure contract tests (Node built-in runner).
 *
 * Verifies the subtraction refactor of the result page: final chart line-ups,
 * deleted modules staying deleted, curve statistics living only in the loss
 * chart, the per-model third summary card, the trailing intercept column and
 * the absence of removed disclaimer wording in production code. Source files
 * are scanned as text; pure helpers are imported for behaviour checks.
 * Run with:
 *   node --test tests/frontend/result_page_structure.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { UNRECORDED, formatNumber, modelScaleSummary } from '../../frontend/src/utils/training.js'
import { summarizeMulticlassCoefficients } from '../../frontend/src/utils/modelExplanation.js'

const SRC_ROOT = fileURLToPath(new URL('../../frontend/src/', import.meta.url))
const RESULT_DIR = `${SRC_ROOT}components/result/`

const readSource = (relativePath) => readFileSync(`${SRC_ROOT}${relativePath}`, 'utf8')
const readResultComponent = (name) => readSource(`components/result/${name}`)

const resultComponentSources = () =>
  readdirSync(RESULT_DIR)
    .filter((file) => file.endsWith('.vue'))
    .map((file) => readFileSync(`${RESULT_DIR}${file}`, 'utf8'))

const productionSources = () => [
  ...resultComponentSources(),
  readSource('utils/training.js'),
  readSource('utils/modelExplanation.js'),
  readSource('utils/resultDiagnostics.js'),
]

// --- 1. classification tab keeps exactly four charts -------------------------

test('classification performance renders exactly the four agreed charts', () => {
  const source = readResultComponent('ClassificationPerformance.vue')

  for (const component of [
    'PerClassMetricsChart',
    'ConfusionMatrixChart',
    'RocCurveChart',
    'PredictionConfidenceChart',
  ]) {
    assert.ok(source.includes(`import ${component}`), `${component} must be imported`)
  }

  const imports = source.match(/^import \w+ from '\.\/\w+\.vue'$/gm) ?? []
  assert.equal(imports.length, 4, 'no other result component may be imported')
})

// --- 2. regression tab keeps exactly four charts ----------------------------

test('regression performance renders exactly the four agreed charts', () => {
  const source = readResultComponent('RegressionPerformance.vue')

  for (const component of [
    'ActualPredictedChart',
    'ResidualScatterChart',
    'ResidualHistogramChart',
    'TopAbsoluteErrorsChart',
  ]) {
    assert.ok(source.includes(`import ${component}`), `${component} must be imported`)
  }

  const imports = source.match(/^import \w+ from '\.\/\w+\.vue'$/gm) ?? []
  assert.equal(imports.length, 4, 'no other result component may be imported')
})

// --- 3. TrainingProcess no longer references the deleted modules ------------

test('training process drops the split donut, curve summary card and parameter table', () => {
  const source = readResultComponent('TrainingProcess.vue')

  for (const removed of [
    'TrainingSplitChart',
    'TrainingCurveSummary',
    'EffectiveParametersTable',
  ]) {
    assert.ok(!source.includes(removed), `${removed} must not be referenced`)
  }
})

// --- 4. non-iterative models never get a (simulated) loss chart -------------

test('the loss chart is only rendered for iterative models', () => {
  const source = readResultComponent('TrainingProcess.vue')

  assert.ok(
    /<TrainingLossChart[^>]*v-if="iterative"/s.test(source),
    'TrainingLossChart must be guarded by the iterative check',
  )
  assert.ok(source.includes('isIterativeModel'), 'the iterative check must come from the helper')
})

// --- 5. curve statistics live only inside the loss chart ---------------------

test('curve statistics are compact tags of the loss chart, not a separate card', () => {
  const processSource = readResultComponent('TrainingProcess.vue')
  assert.ok(!processSource.includes('curveSummary'), 'TrainingProcess must not summarize curves')
  assert.ok(!processSource.includes('summarizeTrainingCurve'))

  const chartSource = readResultComponent('TrainingLossChart.vue')
  assert.ok(chartSource.includes('curveSummary'), 'TrainingLossChart owns the compact statistics')
  for (const label of ['记录轮数', '初始值', '最终值', '最小值']) {
    assert.ok(chartSource.includes(label), `missing compact tag: ${label}`)
  }
})

// --- 6. the third summary card is model-aware --------------------------------

test('modelScaleSummary picks the per-model scale description', () => {
  assert.deepEqual(
    modelScaleSummary('logistic_regression', { n_iter: 10 }, { max_iter: 100 }, null),
    { label: '迭代情况', value: '10 / 100' },
  )
  assert.deepEqual(
    modelScaleSummary('knn_classifier', {}, { n_neighbors: 5 }, null),
    { label: '邻居数量 K', value: '5' },
  )
  assert.deepEqual(
    modelScaleSummary('gaussian_nb', { classes: [0, 1, 2] }, null, null),
    { label: '类别数', value: '3' },
  )
  assert.deepEqual(
    modelScaleSummary('decision_tree_regressor', {}, { max_depth: 8 }, null),
    { label: '最大深度', value: '8' },
  )
  assert.deepEqual(
    modelScaleSummary('random_forest_regressor', {}, null, { n_estimators: 200 }),
    { label: '树数量', value: '200' },
  )
  assert.deepEqual(
    modelScaleSummary('unknown_model', {}, null, null),
    { label: '模型规模', value: UNRECORDED },
  )
})

// --- 7./8./9. trailing intercept column of the multiclass heatmap ------------

test('a valid per-class intercept becomes the last heatmap column', () => {
  const result = summarizeMulticlassCoefficients({
    coefficients: [
      [1, 2],
      [3, 4],
    ],
    featureNames: null,
    classes: ['a', 'b'],
    intercept: [0.1, 0.2],
  })

  assert.equal(result.interceptAppended, true)
  assert.deepEqual(result.heatmapMatrix, [
    [1, 2, 0.1],
    [3, 4, 0.2],
  ])

  const component = readResultComponent('MulticlassCoefficientHeatmap.vue')
  assert.ok(component.includes('heatmapMatrix'), 'the heatmap must use the extended matrix')
  assert.ok(component.includes('截距'), 'the appended column needs its label')
})

test('a shape-mismatched intercept is not appended', () => {
  const result = summarizeMulticlassCoefficients({
    coefficients: [
      [1, 2],
      [3, 4],
    ],
    featureNames: null,
    classes: ['a', 'b'],
    intercept: [0.1],
  })

  assert.equal(result.interceptAppended, false)
  assert.deepEqual(result.heatmapMatrix, [
    [1, 2],
    [3, 4],
  ])
})

test('extending the heatmap never modifies the input matrix or intercept', () => {
  const coefficients = Object.freeze([
    Object.freeze([1, 2]),
    Object.freeze([3, 4]),
  ])
  const intercept = Object.freeze([0.1, 0.2])

  const result = summarizeMulticlassCoefficients({
    coefficients,
    featureNames: null,
    classes: ['a', 'b'],
    intercept,
  })

  assert.equal(result.interceptAppended, true)
  assert.deepEqual(coefficients, [
    [1, 2],
    [3, 4],
  ])
  assert.deepEqual(intercept, [0.1, 0.2])
})

// --- 10./11. removed disclaimer wording stays removed -------------------------

const REMOVED_PHRASES = [
  '解释说明',
  '重要性说明',
  '系数说明',
  '训练过程说明',
  '分类结果解读',
  '回归结果解读',
  '为什么没有损失曲线',
  '不代表因果关系',
  '不代表特征选择结论',
  '机制说明不代表',
  '本次训练记录',
  '统计量说明',
  '模型说明',
  '结构说明',
  '当前接口未保存',
  '不代表变量之间存在因果关系',
]

test('removed explanatory wording does not appear anywhere in production code', () => {
  const failures = []
  for (const source of productionSources()) {
    for (const phrase of REMOVED_PHRASES) {
      if (source.includes(phrase)) failures.push(phrase)
    }
  }
  assert.deepEqual(failures, [], `removed wording still present: ${failures.join(', ')}`)
})

// --- 12. no NaN / undefined leaks ---------------------------------------------

test('formatting helpers fall back to 未记录 instead of NaN or undefined text', () => {
  for (const value of [null, undefined, Number.NaN, Number.POSITIVE_INFINITY, '']) {
    const text = formatNumber(value)
    assert.equal(text, UNRECORDED)
    assert.ok(!text.includes('NaN'))
    assert.ok(!text.includes('undefined'))
  }

  assert.equal(modelScaleSummary('knn_classifier', {}, null, null).value, UNRECORDED)
})

// --- layout guards --------------------------------------------------------------

test('result layouts never use the removed 2fr/1fr split', () => {
  for (const source of resultComponentSources()) {
    assert.ok(!source.includes('2fr'), 'found a 2fr track in a result component')
  }
})

test('equal grids stretch their children so cards align top and bottom', () => {
  for (const name of ['ClassificationPerformance.vue', 'RegressionPerformance.vue', 'GaussianNBExplanation.vue']) {
    const source = readResultComponent(name)
    assert.ok(source.includes('align-items: stretch'), `${name} must stretch row items`)
    assert.ok(source.includes('repeat(2, minmax(0, 1fr))'), `${name} must use equal columns`)
  }
})
