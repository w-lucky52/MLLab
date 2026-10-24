/**
 * Phase 6A — result diagnostics contract tests (Node built-in test runner).
 *
 * Covers the pure helpers behind the new classification distribution /
 * confidence charts and the regression residual diagnostics. Run with:
 *   node --test tests/frontend/result_diagnostics.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildBins,
  classDistribution,
  confidenceHistogram,
  predictionConfidence,
  regressionDiagnostics,
  regressionInsight,
} from '../../frontend/src/utils/resultDiagnostics.js'

// --- classification: class distribution -------------------------------------

test('class distribution keeps scoreClasses order and counts each class', () => {
  const yTrue = [2, 2, 0, 1]
  const yPred = [2, 0, 0, 1]
  const result = classDistribution(yTrue, yPred, [2, 0, 1], null)

  assert.equal(result.available, true)
  assert.deepEqual(
    result.classes.map((item) => item.value),
    [2, 0, 1],
  )
  assert.deepEqual(
    result.classes.map((item) => item.actualCount),
    [2, 1, 1],
  )
  assert.deepEqual(
    result.classes.map((item) => item.predictedCount),
    [1, 2, 1],
  )
})

test('actual and predicted totals always equal the sample counts', () => {
  const yTrue = ['a', 'b', 'b', 'c']
  const yPred = ['a', 'a', 'c', 'c']
  const result = classDistribution(yTrue, yPred, null, null)

  assert.equal(result.actualTotal, yTrue.length)
  assert.equal(result.predictedTotal, yPred.length)
  assert.equal(
    result.classes.reduce((sum, item) => sum + item.actualCount, 0),
    yTrue.length,
  )
  assert.equal(
    result.classes.reduce((sum, item) => sum + item.predictedCount, 0),
    yPred.length,
  )
})

test('classes missing from scoreClasses are appended in first-appearance order', () => {
  const result = classDistribution(['x', 'y', 'x'], ['y', 'y', 'z'], ['y'], null)

  assert.deepEqual(
    result.classes.map((item) => item.value),
    ['y', 'x', 'z'],
  )
})

test('friendly names are used, invalid names fall back to raw labels', () => {
  const friendly = classDistribution([0, 1], [0, 1], [0, 1], ['setosa', 'versicolor'])
  assert.deepEqual(
    friendly.classes.map((item) => item.name),
    ['setosa', 'versicolor'],
  )

  const fallback = classDistribution([0, 1], [0, 1], [0, 1], ['only-one'])
  assert.deepEqual(
    fallback.classes.map((item) => item.name),
    ['0', '1'],
  )

  const emptyName = classDistribution([0, 1], [0, 1], [0, 1], ['setosa', ''])
  assert.deepEqual(
    emptyName.classes.map((item) => item.name),
    ['0', '1'],
  )
})

test('numeric and string labels are never merged into one bucket', () => {
  const result = classDistribution([1, '1'], [1, '1'], null, null)

  assert.equal(result.classes.length, 2)
  assert.deepEqual(
    result.classes.map((item) => item.value),
    [1, '1'],
  )
})

test('invalid class inputs produce an unavailable result', () => {
  assert.equal(classDistribution([], [], [0], null).available, false)
  assert.equal(classDistribution([0, 1], [0], [0, 1], null).available, false)
  assert.equal(classDistribution([[0], [1]], [0, 1], null, null).available, false)
  assert.equal(classDistribution([0, Number.NaN], [0, 1], null, null).available, false)
  assert.equal(classDistribution([0, Infinity], [0, 1], null, null).available, false)
  assert.equal(classDistribution(null, [0], null, null).available, false)
})

// --- classification: confidence / margin ------------------------------------

test('probability confidence is the row maximum', () => {
  const result = predictionConfidence(
    [0, 1],
    [0, 1],
    [
      [0.8, 0.2],
      [0.1, 0.9],
    ],
    'probability',
  )

  assert.equal(result.available, true)
  assert.equal(result.kind, 'probability')
  assert.deepEqual(result.values, [0.8, 0.9])
})

test('probability rows must sum to one and stay inside [0, 1]', () => {
  const result = predictionConfidence(
    [0],
    [0],
    [[0.7, 0.5]],
    'probability',
  )
  assert.equal(result.available, false)

  const outside = predictionConfidence([0], [0], [[1.2, -0.2]], 'probability')
  assert.equal(outside.available, false)
})

test('decision scores produce the top1 minus top2 margin', () => {
  const result = predictionConfidence(
    [0, 1],
    [0, 1],
    [
      [2.5, 0.5],
      [-1.0, 3.0],
    ],
    'decision',
  )

  assert.equal(result.available, true)
  assert.equal(result.kind, 'decision')
  assert.deepEqual(result.values, [2, 4])
})

test('tied decision scores produce a zero margin', () => {
  const result = predictionConfidence([0], [0], [[1.5, 1.5, 0.0]], 'decision')

  assert.equal(result.available, true)
  assert.deepEqual(result.values, [0])
})

test('decision scores are rejected as probability-free probabilities', () => {
  const oneClass = predictionConfidence([0], [0], [[1.0]], 'decision')
  assert.equal(oneClass.available, false)

  const badRow = predictionConfidence([0, 1], [0, 1], [[1, 2]], 'decision')
  assert.equal(badRow.available, false)

  const nonFinite = predictionConfidence([0], [0], [[1, Number.NaN]], 'decision')
  assert.equal(nonFinite.available, false)
})

test('correct and wrong samples follow y_true vs y_pred order', () => {
  const result = predictionConfidence(
    ['a', 'b', 'c'],
    ['a', 'c', 'c'],
    [
      [0.9, 0.05, 0.05],
      [0.1, 0.2, 0.7],
      [0.1, 0.1, 0.8],
    ],
    'probability',
  )

  assert.deepEqual(result.correct, [true, false, true])
  assert.equal(result.correctCount, 2)
  assert.equal(result.wrongCount, 1)
})

test('confidence histogram places every sample in exactly one bin', () => {
  const result = confidenceHistogram(
    [0, 0, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [
      [0.9, 0.1],
      [0.4, 0.6],
      [0.2, 0.8],
      [0.7, 0.3],
      [0.5, 0.5],
    ],
    'probability',
  )

  assert.equal(result.available, true)
  assert.equal(result.bins.length, 10)
  assert.equal(
    result.bins.reduce((sum, bin) => sum + bin.count, 0),
    result.total,
  )
  assert.equal(
    result.bins.reduce((sum, bin) => sum + bin.correct + bin.wrong, 0),
    result.total,
  )
  assert.equal(result.correctCount + result.wrongCount, result.total)
})

test('constant confidence values still produce one valid bin', () => {
  const result = confidenceHistogram(
    [0, 0],
    [0, 0],
    [
      [0.5, 0.3, 0.2],
      [0.5, 0.25, 0.25],
    ],
    'probability',
  )

  assert.equal(result.available, true)
  assert.equal(result.bins.length, 1)
  assert.equal(result.bins[0].count, 2)
  assert.equal(result.bins[0].min, 0.5)
  assert.equal(result.bins[0].max, 0.5)
})

test('histogram excludes unusable inputs and keeps the maximum edge', () => {
  assert.equal(buildBins([], 10).available, false)
  assert.equal(buildBins([1, Number.POSITIVE_INFINITY], 10).available, false)

  const edges = buildBins([0, 0.5, 1], 10)
  assert.equal(edges.available, true)
  assert.equal(
    edges.bins.reduce((sum, bin) => sum + bin.count, 0),
    3,
  )
  assert.equal(edges.bins.at(-1).count >= 1, true)
})

test('confidence inputs with mismatched shapes are unavailable', () => {
  assert.equal(predictionConfidence([], [], [], 'probability').available, false)
  assert.equal(predictionConfidence([0, 1], [0], [[1]], 'probability').available, false)
  assert.equal(
    predictionConfidence([0, 1], [0, 1], [[0.5, 0.5], [0.5]], 'probability').available,
    false,
  )
  assert.equal(
    predictionConfidence([0], [0], [[0.5, 'x']], 'probability').available,
    false,
  )
})

// --- regression diagnostics -------------------------------------------------

test('residual keeps the true-minus-predicted sign', () => {
  const result = regressionDiagnostics([5, 1], [3, 1])

  assert.deepEqual(result.residuals, [2, 0])
  assert.deepEqual(result.absolute_errors, [2, 0])
  assert.deepEqual(result.residual_points, [
    { index: 0, sample: 1, actual: 5, predicted: 3, residual: 2 },
    { index: 1, sample: 2, actual: 1, predicted: 1, residual: 0 },
  ])
})

test('mean residual and median absolute error are computed from real values', () => {
  const result = regressionDiagnostics([1, 2, 3, 4], [2, 2, 5, 1])

  assert.deepEqual(result.residuals, [-1, 0, -2, 3])
  assert.equal(result.mean_residual, 0)
  assert.deepEqual(result.absolute_errors, [1, 0, 2, 3])
  assert.equal(result.median_absolute_error, 1.5)
  assert.equal(result.max_absolute_error, 3)
})

test('top absolute errors are sorted descending with stable ties', () => {
  const yTrue = Array.from({ length: 12 }, (_, index) => index + 10)
  const yPred = yTrue.map((value, index) => value - (1 + (index % 3)))

  const result = regressionDiagnostics(yTrue, yPred)
  const top = result.top_absolute_errors

  assert.equal(top.length, 10)
  assert.deepEqual(
    top.map((item) => item.absoluteError),
    [3, 3, 3, 3, 2, 2, 2, 2, 1, 1],
  )
  for (let index = 1; index < top.length; index += 1) {
    const previous = top[index - 1]
    const current = top[index]
    assert.equal(previous.absoluteError >= current.absoluteError, true)
    if (previous.absoluteError === current.absoluteError) {
      assert.equal(previous.index < current.index, true)
    }
  }
})

test('sample numbers are 1-based test-set positions', () => {
  const result = regressionDiagnostics([1, 2, 3], [1, 1, 1])

  assert.deepEqual(
    result.actual_predicted_points.map((point) => point.sample),
    [1, 2, 3],
  )
  assert.deepEqual(
    result.top_absolute_errors.map((item) => item.sample),
    [3, 2, 1],
  )
})

test('residual histogram keeps every sample and handles constant residuals', () => {
  const result = regressionDiagnostics([1, 1, 1], [1, 1, 1])
  const histogram = buildBins(result.residuals, 10)

  assert.equal(histogram.available, true)
  assert.equal(histogram.bins.length, 1)
  assert.equal(histogram.bins[0].count, 3)
  assert.equal(histogram.total, result.sample_count)

  const varied = regressionDiagnostics([1, 2, 3, 4], [0, 0, 0, 0])
  const bins = buildBins(varied.residuals, 10)
  assert.equal(
    bins.bins.reduce((sum, bin) => sum + bin.count, 0),
    varied.sample_count,
  )
})

test('regression inputs with empty, mismatched or non-finite values are unavailable', () => {
  assert.equal(regressionDiagnostics([], []).available, false)
  assert.equal(regressionDiagnostics([1, 2], [1]).available, false)
  assert.equal(regressionDiagnostics([1, Number.NaN], [1, 2]).available, false)
  assert.equal(regressionDiagnostics([1, 2], [1, Number.POSITIVE_INFINITY]).available, false)
  assert.equal(regressionDiagnostics(null, [1]).available, false)
  assert.equal(buildBins([Number.NaN], 10).available, false)
})

test('input arrays are never modified', () => {
  const yTrue = [3, 1, 2]
  const yPred = [2, 2, 2]
  const trueCopy = [...yTrue]
  const predCopy = [...yPred]

  classDistribution(['b', 'a'], ['a', 'a'], ['a', 'b'], ['A', 'B'])
  predictionConfidence([0, 1], [0, 1], [[0.9, 0.1], [0.2, 0.8]], 'probability')
  regressionDiagnostics(yTrue, yPred)

  assert.deepEqual(yTrue, trueCopy)
  assert.deepEqual(yPred, predCopy)
})

test('regression insight reports direction without causal claims', () => {
  const under = regressionInsight(regressionDiagnostics([5, 5, 5], [3, 3, 3]))
  assert.equal(under.available, true)
  assert.match(under.statements[0], /低估/)
  assert.doesNotMatch(under.statements.join(''), /很好|很差|过拟合|异方差|最优/)

  const over = regressionInsight(regressionDiagnostics([3, 3, 3], [5, 5, 5]))
  assert.match(over.statements[0], /高估/)

  const near = regressionInsight(regressionDiagnostics([100, 101], [100.0001, 101.0]))
  assert.match(near.statements[0], /接近 0/)
})
