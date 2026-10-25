/**
 * Phase 7A — model explanation data contract tests (Node built-in runner).
 *
 * Covers the pure helpers behind the 模型解释 tab after the subtraction
 * refactor: coefficient rows with stable ordering, the multiclass heatmap
 * matrix with an optional trailing intercept column, feature importance rows,
 * the GaussianNB mean/variance pair, MLP dimensions, the KNN workflow, input
 * immutability and the objective wording of all user-visible reasons.
 * Run with:
 *   node --test tests/frontend/model_explanation.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { UNRECORDED } from '../../frontend/src/utils/training.js'
import {
  gaussianExplanationData,
  interceptColumn,
  knnExplanationData,
  summarizeCoefficients,
  summarizeFeatureImportances,
  summarizeMLPArchitecture,
  summarizeMulticlassCoefficients,
} from '../../frontend/src/utils/modelExplanation.js'

const snapshot = (value) => (value === null ? null : JSON.parse(JSON.stringify(value)))

// --- summarizeCoefficients ----------------------------------------------------

test('coefficient rows are sorted by magnitude with signed values kept', () => {
  const result = summarizeCoefficients({
    coefficients: [2, -1, 0, 5],
    featureNames: null,
    intercept: 3,
  })

  assert.equal(result.available, true)
  assert.equal(result.featureCount, 4)
  assert.deepEqual(
    result.rows.map((row) => row.value),
    [5, 2, -1, 0],
  )
  assert.equal(result.rows[0].feature, '特征 4')
  assert.equal(result.intercept, 3)
})

test('equal magnitudes keep the original feature order (stable sort)', () => {
  const result = summarizeCoefficients({
    coefficients: [3, -3, 1],
    featureNames: null,
    intercept: null,
  })

  assert.deepEqual(
    result.rows.map((row) => row.value),
    [3, -3, 1],
  )
  assert.equal(result.rows[0].feature, '特征 1')
  assert.equal(result.rows[1].feature, '特征 2')
  assert.equal(result.intercept, null)
})

test('missing feature names fall back to positional labels for regression', () => {
  const result = summarizeCoefficients({
    coefficients: [1, 2],
    featureNames: ['a', 'b', 'c'],
    intercept: null,
  })

  assert.equal(result.available, true)
  // Rows are ordered by magnitude, so value 2 (特征 2) comes first.
  assert.equal(result.rows[0].feature, '特征 2')
  assert.equal(result.rows[0].value, 2)
  assert.equal(result.rows[1].feature, '特征 1')
  assert.equal(result.featureNames.mismatched, true)
})

test('non-finite or empty coefficients are rejected with a reason', () => {
  for (const coefficients of [null, [], [1, NaN], [1, Infinity], 'nope']) {
    const result = summarizeCoefficients({ coefficients, featureNames: null, intercept: null })
    assert.equal(result.available, false)
    assert.ok(result.reason.length > 0)
  }
})

test('intercept accepts a number or a single-element array', () => {
  const single = summarizeCoefficients({
    coefficients: [1],
    featureNames: null,
    intercept: [2.5],
  })
  assert.equal(single.intercept, 2.5)

  const missing = summarizeCoefficients({
    coefficients: [1],
    featureNames: null,
    intercept: [1, 2],
  })
  assert.equal(missing.intercept, null)
})

// --- summarizeMulticlassCoefficients ------------------------------------------

test('multiclass summary keeps matrix shape as class and feature counts', () => {
  const result = summarizeMulticlassCoefficients({
    coefficients: [
      [1, -4, 2],
      [0, 3, -3],
    ],
    featureNames: null,
    classes: ['a', 'b'],
    scoreClasses: null,
    intercept: [0.5, -0.5],
  })

  assert.equal(result.available, true)
  assert.equal(result.classCount, 2)
  assert.equal(result.featureCount, 3)
  assert.deepEqual(result.classNames.names, ['a', 'b'])
})

test('a per-class intercept is appended as the last heatmap column', () => {
  const result = summarizeMulticlassCoefficients({
    coefficients: [
      [1, -4, 2],
      [0, 3, -3],
    ],
    featureNames: null,
    classes: ['a', 'b'],
    scoreClasses: null,
    intercept: [0.5, -0.5],
  })

  assert.equal(result.interceptAppended, true)
  assert.deepEqual(result.heatmapMatrix, [
    [1, -4, 2, 0.5],
    [0, 3, -3, -0.5],
  ])
  // The coefficient matrix itself stays untouched.
  assert.deepEqual(result.matrix, [
    [1, -4, 2],
    [0, 3, -3],
  ])
})

test('a mismatched or invalid intercept is never appended', () => {
  const base = {
    coefficients: [
      [1, -4, 2],
      [0, 3, -3],
    ],
    featureNames: null,
    classes: ['a', 'b'],
    scoreClasses: null,
  }

  for (const intercept of [[0.5], [0.5, -0.5, 1], [0.5, Number.NaN], 0.5, null]) {
    const result = summarizeMulticlassCoefficients({ ...base, intercept })
    assert.equal(result.interceptAppended, false, JSON.stringify(intercept))
    assert.deepEqual(result.heatmapMatrix, base.coefficients, JSON.stringify(intercept))
  }
})

test('ragged or non-finite multiclass matrices are rejected', () => {
  const ragged = summarizeMulticlassCoefficients({
    coefficients: [[1, 2], [3]],
    featureNames: null,
    classes: null,
    scoreClasses: null,
    intercept: null,
  })
  assert.equal(ragged.available, false)

  const nonFinite = summarizeMulticlassCoefficients({
    coefficients: [[1, NaN]],
    featureNames: null,
    classes: null,
    scoreClasses: null,
    intercept: null,
  })
  assert.equal(nonFinite.available, false)
})

test('interceptColumn only accepts an exact-length finite vector', () => {
  assert.deepEqual(interceptColumn([1, -2], 2), [1, -2])
  assert.equal(interceptColumn([1, -2], 3), null)
  assert.equal(interceptColumn([1], 2), null)
  assert.equal(interceptColumn([1, Number.NaN], 2), null)
  assert.equal(interceptColumn(1, 2), null)
  assert.equal(interceptColumn(null, 2), null)
})

// --- summarizeFeatureImportances ----------------------------------------------

test('importance rows are sorted descending with derived percents', () => {
  const result = summarizeFeatureImportances({
    featureImportances: [0.5, 0.25, 0.125, 0.125],
    featureNames: ['f1', 'f2', 'f3', 'f4'],
  })

  assert.equal(result.available, true)
  assert.equal(result.featureCount, 4)
  assert.deepEqual(
    result.rows.map((row) => row.feature),
    ['f1', 'f2', 'f3', 'f4'],
  )
  assert.ok(Math.abs(result.rows[0].percent - 50) < 1e-9)
})

test('zero total importance disables every derived percentage', () => {
  const result = summarizeFeatureImportances({
    featureImportances: [0, 0, 0],
    featureNames: null,
  })

  assert.equal(result.available, true)
  assert.ok(result.rows.every((row) => row.percent === null))
  assert.equal(result.rows[0].feature, '特征 1')
})

test('negative, non-finite or length-mismatched importances are rejected', () => {
  const negative = summarizeFeatureImportances({
    featureImportances: [0.5, -0.1],
    featureNames: null,
  })
  assert.equal(negative.available, false)
  assert.ok(negative.reason.includes('负数'))

  for (const values of [[NaN, 0.5], [0.5, Infinity]]) {
    const nonFinite = summarizeFeatureImportances({ featureImportances: values, featureNames: null })
    assert.equal(nonFinite.available, false)
  }

  const mismatched = summarizeFeatureImportances({
    featureImportances: [0.5, 0.5],
    featureNames: ['f1', 'f2', 'f3'],
  })
  assert.equal(mismatched.available, false)
  assert.ok(mismatched.reason.includes('不一致'))
})

// --- gaussianExplanationData ----------------------------------------------------

test('GaussianNB data exposes the validated mean/variance pair', () => {
  const result = gaussianExplanationData({
    trainingInfo: {
      classes: ['a', 'b'],
      class_feature_means: [
        [1, 2, 3],
        [4, 5, 6],
      ],
      class_feature_variances: [
        [0.5, 2, 9],
        [1.5, 0.25, 4],
      ],
      n_features_in: 3,
    },
    featureNames: null,
    scoreClasses: null,
  })

  assert.notEqual(result, null)
  assert.equal(result.rowCount, 2)
  assert.equal(result.columnCount, 3)
  assert.deepEqual(result.classes.names, ['a', 'b'])
})

test('GaussianNB rejects shape mismatch and negative variance', () => {
  const mismatched = gaussianExplanationData({
    trainingInfo: {
      classes: ['a', 'b'],
      class_feature_means: [[1, 2]],
      class_feature_variances: [
        [1, 2],
        [3, 4],
      ],
    },
    featureNames: null,
    scoreClasses: null,
  })
  assert.equal(mismatched, null)

  const negative = gaussianExplanationData({
    trainingInfo: {
      classes: ['a'],
      class_feature_means: [[1]],
      class_feature_variances: [[-0.5]],
    },
    featureNames: null,
    scoreClasses: null,
  })
  assert.equal(negative, null)
})

// --- summarizeMLPArchitecture ---------------------------------------------------

test('MLP summary exposes the recorded network dimensions', () => {
  const result = summarizeMLPArchitecture({
    n_features_in: 4,
    classes: [0, 1],
    resolved_parameters: { hidden_size: 3 },
  })

  assert.equal(result.inputText, '4')
  assert.equal(result.hiddenText, '3')
  assert.equal(result.outputText, '2')
  assert.equal(result.structureText, '4 → 3 → 2')
})

test('MLP summary falls back to 未记录 when fields are missing', () => {
  const result = summarizeMLPArchitecture({})

  assert.equal(result.inputText, UNRECORDED)
  assert.equal(result.hiddenText, UNRECORDED)
  assert.equal(result.outputText, UNRECORDED)
  assert.equal(result.structureText, `${UNRECORDED} → ${UNRECORDED} → ${UNRECORDED}`)
})

// --- knnExplanationData -----------------------------------------------------------

test('KNN workflow mentions the recorded K and weights without inventing values', () => {
  const recorded = knnExplanationData({
    resolved_parameters: { n_neighbors: 5, weights: 'distance' },
  })
  assert.equal(recorded.workflow.length, 3)
  assert.ok(recorded.workflow.some((step) => step.includes('K = 5')))
  assert.ok(recorded.workflow.some((step) => step.includes('distance')))

  const missing = knnExplanationData({})
  assert.ok(missing.workflow.some((step) => step.includes(`K = ${UNRECORDED}`)))
  assert.ok(missing.workflow.some((step) => step.includes(UNRECORDED)))
})

// --- input immutability ----------------------------------------------------------

test('every helper leaves its inputs untouched', () => {
  const coefficients = [2, -1, 0, 5]
  const matrix = [
    [1, -4, 2],
    [0, 3, -3],
  ]
  const intercept = [0.5, -0.5]
  const importances = [0.5, 0.25, 0.125, 0.125]
  const names = ['f1', 'f2', 'f3', 'f4']
  const gnbInfo = {
    classes: ['a', 'b'],
    class_feature_means: [
      [1, 2, 3],
      [4, 5, 6],
    ],
    class_feature_variances: [
      [0.5, 2, 9],
      [1.5, 0.25, 4],
    ],
  }
  const mlpInfo = {
    n_features_in: 4,
    classes: [0, 1],
    resolved_parameters: { hidden_size: 3 },
  }
  const knnInfo = { resolved_parameters: { n_neighbors: 5, weights: 'uniform' } }

  const inputSnapshot = {
    coefficients: snapshot(coefficients),
    matrix: snapshot(matrix),
    intercept: snapshot(intercept),
    importances: snapshot(importances),
    names: snapshot(names),
    gnbInfo: snapshot(gnbInfo),
    mlpInfo: snapshot(mlpInfo),
    knnInfo: snapshot(knnInfo),
  }

  summarizeCoefficients({ coefficients, featureNames: names, intercept: 1 })
  summarizeMulticlassCoefficients({ coefficients: matrix, featureNames: names, classes: null, intercept })
  summarizeFeatureImportances({ featureImportances: importances, featureNames: names })
  gaussianExplanationData({ trainingInfo: gnbInfo, featureNames: null, scoreClasses: null })
  summarizeMLPArchitecture(mlpInfo)
  knnExplanationData(knnInfo)

  assert.deepEqual(coefficients, inputSnapshot.coefficients)
  assert.deepEqual(matrix, inputSnapshot.matrix)
  assert.deepEqual(intercept, inputSnapshot.intercept)
  assert.deepEqual(importances, inputSnapshot.importances)
  assert.deepEqual(names, inputSnapshot.names)
  assert.deepEqual(gnbInfo, inputSnapshot.gnbInfo)
  assert.deepEqual(mlpInfo, inputSnapshot.mlpInfo)
  assert.deepEqual(knnInfo, inputSnapshot.knnInfo)
})

// --- objective wording ------------------------------------------------------------

const FORBIDDEN_WORDS = ['最好', '很差', '泛化良好', '最优模型', '最重要']

test('rejection reasons never contain subjective or causal claims', () => {
  const texts = []

  const rejections = [
    summarizeCoefficients({ coefficients: [], featureNames: null, intercept: null }).reason,
    summarizeMulticlassCoefficients({ coefficients: [1], featureNames: null, classes: null }).reason,
    summarizeFeatureImportances({ featureImportances: [-1], featureNames: null }).reason,
    summarizeFeatureImportances({
      featureImportances: [0.5, 0.5],
      featureNames: ['a', 'b', 'c'],
    }).reason,
  ]
  texts.push(...rejections.filter(Boolean))
  texts.push(...knnExplanationData({}).workflow)

  const joined = texts.join('\n')
  for (const word of FORBIDDEN_WORDS) {
    assert.ok(!joined.includes(word), `unexpected subjective wording: ${word}`)
  }
  assert.ok(!joined.includes('因果'), 'causal wording found')
})
