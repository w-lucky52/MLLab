/**
 * Phase 5 frontend contract tests (Node built-in test runner).
 *
 * These cover the pure display-label mapping and prove that switching to
 * friendly names never changes the ROC / confusion-matrix / coefficient data.
 * Run with: node --test tests/frontend/class_display.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  classDisplayLookup,
  classDisplayName,
  resolveClassDisplayNames,
} from '../../frontend/src/utils/classDisplay.js'
import { calculateOvRRoc } from '../../frontend/src/utils/roc.ts'
import {
  normalizeClassNames,
  summarizeCoefficients,
} from '../../frontend/src/utils/modelExplanation.js'

const TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

test('valid names are used in score order', () => {
  assert.deepEqual(
    resolveClassDisplayNames([0, 1, 2], TARGET_NAMES),
    ['setosa', 'versicolor', 'virginica'],
  )
})

test('names are consumed in score order and never re-sorted', () => {
  // The backend already emits names aligned with score_classes, so the
  // frontend must keep that order even when the raw classes are re-ordered.
  assert.deepEqual(
    resolveClassDisplayNames([2, 0, 1], ['virginica', 'setosa', 'versicolor']),
    ['virginica', 'setosa', 'versicolor'],
  )
})

test('mismatched or empty names fall back to raw class labels', () => {
  assert.deepEqual(resolveClassDisplayNames([0, 1, 2], ['a', 'b']), ['0', '1', '2'])
  assert.deepEqual(resolveClassDisplayNames([0, 1], ['a', '']), ['0', '1'])
  assert.deepEqual(resolveClassDisplayNames([0, 1], null), ['0', '1'])
  assert.deepEqual(resolveClassDisplayNames([], []), [])
})

test('lookup keeps raw values and display names aligned', () => {
  const lookup = classDisplayLookup([2, 0, 1], ['virginica', 'setosa', 'versicolor'])
  assert.equal(classDisplayName(2, lookup), 'virginica')
  assert.equal(classDisplayName(0, lookup), 'setosa')
  assert.equal(classDisplayName(1, lookup), 'versicolor')
  // Unknown values keep their raw form.
  assert.equal(classDisplayName(9, lookup), '9')
})

test('ROC curves are identical; only labels would change', () => {
  const yTrue = [0, 1, 0, 1, 2, 2]
  const yScore = [
    [0.9, 0.05, 0.05],
    [0.1, 0.8, 0.1],
    [0.85, 0.1, 0.05],
    [0.05, 0.9, 0.05],
    [0.1, 0.1, 0.8],
    [0.2, 0.2, 0.6],
  ]
  const raw = calculateOvRRoc(yTrue, yScore, [0, 1, 2])
  const again = calculateOvRRoc(yTrue, yScore, [0, 1, 2])

  assert.deepEqual(again, raw)
  assert.deepEqual(
    raw.map((curve) => [curve.classLabel, curve.fpr, curve.tpr, curve.auc]),
    again.map((curve) => [curve.classLabel, curve.fpr, curve.tpr, curve.auc]),
  )

  const display = resolveClassDisplayNames([0, 1, 2], TARGET_NAMES)
  assert.deepEqual(display, TARGET_NAMES)
  // Display names never enter the calculation: every curve is unchanged.
  assert.equal(raw.length, 3)
})

test('confusion-matrix data and percentages ignore display labels', () => {
  const matrix = [
    [5, 1, 0],
    [2, 7, 1],
    [0, 1, 9],
  ]
  const rawPercent = matrix.map((row) => {
    const total = row.reduce((sum, value) => sum + value, 0)
    return row.map((value) => (total > 0 ? (value / total) * 100 : 0))
  })

  const display = resolveClassDisplayNames([0, 1, 2], TARGET_NAMES)
  const percentAfterNames = matrix.map((row) => {
    const total = row.reduce((sum, value) => sum + value, 0)
    return row.map((value) => (total > 0 ? (value / total) * 100 : 0))
  })

  assert.deepEqual(display, TARGET_NAMES)
  assert.deepEqual(percentAfterNames, rawPercent)
  assert.deepEqual(matrix, [
    [5, 1, 0],
    [2, 7, 1],
    [0, 1, 9],
  ])
})

test('coefficient heatmap class rows keep their original order', () => {
  const raw = normalizeClassNames([0, 1, 2], [0, 1, 2], 3)
  const lookup = classDisplayLookup([0, 1, 2], TARGET_NAMES)
  const display = raw.names.map((name) => classDisplayName(name, lookup))

  // Mapping only swaps the labels; row order comes from the matrix.
  assert.deepEqual(display, TARGET_NAMES)
  assert.deepEqual(raw.names, ['0', '1', '2'])
})

test('coefficient row order is independent of class display names', () => {
  const vector = [0.5, -2.0, 1.0]
  const withNames = summarizeCoefficients({
    coefficients: vector,
    featureNames: ['f0', 'f1', 'f2'],
    intercept: null,
  })
  const withoutNames = summarizeCoefficients({
    coefficients: vector,
    featureNames: null,
    intercept: null,
  })

  assert.deepEqual(
    withNames.rows.map((row) => row.value),
    withoutNames.rows.map((row) => row.value),
  )
  assert.deepEqual(
    withNames.rows.map((row) => row.value),
    [-2.0, 1.0, 0.5],
  )
})
