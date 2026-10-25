/**
 * Phase 7B2B — local model-insight contract tests (Node built-in runner).
 *
 * Covers the pure helpers behind the per-sample explanation charts
 * (contribution rows, KNN neighbours/votes, GaussianNB likelihoods, decision
 * paths, permutation importances) plus source-level guarantees: the sample
 * selector default/reset, Chinese wording, forbidden phrases, the 1fr/1fr
 * layout rules, responsive breakpoints and the absence of new backend
 * requests.
 *
 * Run with:
 *   node --test tests/frontend/explanation_local.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { classDisplayLookup, classDisplayName } from '../../frontend/src/utils/classDisplay.js'
import {
  contributionChartData,
  decisionPathRows,
  explanationSamples,
  gaussianLikelihoodChartData,
  knnNeighborChartData,
  knnVoteChartData,
  normalizeExplanationData,
  permutationImportanceData,
} from '../../frontend/src/utils/modelExplanation.js'

const SRC_ROOT = fileURLToPath(new URL('../../frontend/src/', import.meta.url))
const RESULT_DIR = `${SRC_ROOT}components/result/`
const readResultComponent = (name) => readFileSync(`${RESULT_DIR}${name}`, 'utf8')
const readSource = (relativePath) => readFileSync(`${SRC_ROOT}${relativePath}`, 'utf8')

const NEW_COMPONENTS = [
  'SampleExplanationSelector.vue',
  'LocalFeatureContributionChart.vue',
  'PermutationImportanceChart.vue',
  'GaussianNBLocalLikelihoodChart.vue',
  'DecisionPathExplanation.vue',
  'KNNExplanation.vue',
  'ModelExplanation.vue',
]

const resultComponentSources = () =>
  readdirSync(RESULT_DIR)
    .filter((file) => file.endsWith('.vue'))
    .map((file) => readFileSync(`${RESULT_DIR}${file}`, 'utf8'))

const productionSources = () => [
  ...resultComponentSources(),
  readSource('utils/modelExplanation.js'),
  readSource('utils/resultLabels.js'),
  readSource('utils/training.js'),
  readSource('utils/resultDiagnostics.js'),
]

// --- fixtures -----------------------------------------------------------------

const linearPayload = {
  schema_version: 1,
  kind: 'linear_contribution',
  feature_space: 'standardized',
  feature_names: ['age', 'bmi'],
  sample_count: 1,
  samples: [
    {
      sample_index: 1,
      y_true: 150,
      y_pred: 151.5,
      feature_values: [0.5, -1],
      contributions: [1.5, 50],
      intercept: 100,
      reconstructed_prediction: 151.5,
    },
  ],
}

const knnSample = {
  sample_index: 2,
  y_true: 'a',
  y_pred: 'a',
  neighbors: [
    { rank: 1, train_index: 12, distance: 0.1, label: 'a', vote_weight: 10 },
    { rank: 2, train_index: 3, distance: 0.5, label: 'a', vote_weight: 2 },
    { rank: 3, train_index: 7, distance: 0.9, label: 'b', vote_weight: 1.111 },
  ],
  vote_totals: [
    { class: 'a', weight: 12 },
    { class: 'b', weight: 1.111 },
  ],
}

const knnPayload = {
  schema_version: 1,
  kind: 'knn_neighbors',
  feature_space: 'standardized',
  feature_names: ['f1'],
  sample_count: 1,
  n_neighbors: 3,
  weights: 'distance',
  samples: [knnSample],
}

// --- 1. kind dispatch -----------------------------------------------------------

test('normalizeExplanationData accepts every recorded kind and rejects the rest', () => {
  const base = { schema_version: 1, feature_space: 'standardized', feature_names: ['f1'] }
  const sampleKinds = [
    'linear_contribution',
    'multiclass_linear_contribution',
    'knn_neighbors',
    'gaussian_nb_likelihood',
    'decision_path',
  ]
  for (const kind of sampleKinds) {
    const result = normalizeExplanationData({
      ...base,
      kind,
      samples: [{ sample_index: 1 }],
    })
    assert.equal(result.available, true, kind)
    assert.equal(result.kind, kind)
  }

  const permutation = normalizeExplanationData({
    ...base,
    kind: 'permutation_importance',
    baseline_accuracy: 0.9,
    importances: [0.1],
  })
  assert.equal(permutation.available, true)
  assert.equal(permutation.samples, null)

  for (const bad of [
    null,
    'x',
    { kind: 'unknown_kind', feature_space: 'standardized', feature_names: ['a'] },
    { kind: 'linear_contribution', feature_space: 'raw', feature_names: ['a'], samples: [{ sample_index: 1 }] },
    { kind: 'linear_contribution', feature_space: 'standardized', feature_names: [], samples: [{ sample_index: 1 }] },
    { kind: 'linear_contribution', feature_space: 'standardized', feature_names: ['a'], samples: [] },
    { kind: 'linear_contribution', feature_space: 'standardized', feature_names: ['a'], samples: [{ sample_index: 0 }] },
  ]) {
    assert.equal(normalizeExplanationData(bad).available, false, JSON.stringify(bad))
  }
})

// --- 2./3. sample selector defaults & reset -------------------------------------

test('samples keep their stored order and the first sample is #1', () => {
  const payload = normalizeExplanationData({
    ...knnPayload,
    samples: [
      { ...knnSample, sample_index: 1 },
      { ...knnSample, sample_index: 2 },
    ],
  })
  const samples = explanationSamples(payload)
  assert.equal(samples.length, 2)
  assert.equal(samples[0].sample_index, 1)

  const selector = readResultComponent('SampleExplanationSelector.vue')
  assert.ok(selector.includes('modelValue'), 'the selector is controlled by the parent')
  assert.ok(selector.includes("emit('update:modelValue'"), 'selection changes flow upward')

  const view = readResultComponent('ModelExplanation.vue')
  assert.ok(view.includes('const selectedIndex = ref(0)'), 'default selection is the first sample')
  assert.ok(
    /watch\(\s*\(\) => props\.trainingInfo,\s*\(\) => \{\s*selectedIndex\.value = 0/s.test(view),
    'a new experiment resets the selection to the first sample',
  )
  assert.ok(view.includes('v-if="hasSamples"'), 'the selector hides without valid samples')
})

test('selector labels use the recorded sample index and class display names', () => {
  const selector = readResultComponent('SampleExplanationSelector.vue')
  assert.ok(selector.includes('测试样本 #${sample.sample_index}'))
  assert.ok(selector.includes('classDisplayLookup'))
  assert.ok(selector.includes('真实类别'))
  assert.ok(selector.includes('预测类别'))
  assert.ok(selector.includes('真实值'))
  assert.ok(selector.includes('预测值'))
  assert.ok(selector.includes('解释对象'))
})

// --- 4./5. linear contribution rows ----------------------------------------------

test('contribution rows keep signs and reconstruct the prediction', () => {
  const result = contributionChartData({
    sample: linearPayload.samples[0],
    featureNames: linearPayload.feature_names,
  })
  assert.equal(result.available, true)
  assert.equal(result.rows.length, 2)
  // Sorted by |contribution|: bmi (50) before age (1.5).
  assert.deepEqual(
    result.rows.map((row) => row.feature),
    ['bmi', 'age'],
  )
  assert.equal(result.rows[1].direction, 'positive')
  assert.ok(Math.abs(result.contributionSum + result.intercept - result.reconstructed) < 1e-9)

  const broken = contributionChartData({
    sample: { ...linearPayload.samples[0], reconstructed_prediction: 999 },
    featureNames: linearPayload.feature_names,
  })
  assert.equal(broken.available, false, 'inconsistent reconstruction is rejected')

  const nonFinite = contributionChartData({
    sample: { ...linearPayload.samples[0], contributions: [1, NaN] },
    featureNames: linearPayload.feature_names,
  })
  assert.equal(nonFinite.available, false)
})

test('contribution sorting is stable and never mutates the input sample', () => {
  const sample = {
    sample_index: 1,
    y_true: 0,
    y_pred: 0,
    feature_values: [1, 1, 1],
    contributions: [2, -2, 1],
    intercept: -1,
    reconstructed_prediction: 0,
  }
  const snapshot = JSON.parse(JSON.stringify(sample))
  const result = contributionChartData({ sample, featureNames: ['a', 'b', 'c'] })
  assert.equal(result.available, true)
  // Equal magnitudes keep the original feature order.
  assert.deepEqual(
    result.rows.map((row) => row.feature),
    ['a', 'b', 'c'],
  )
  assert.deepEqual(sample, snapshot)
})

// --- 6. logistic vs SVM titles -----------------------------------------------------

test('logistic regression and linear SVM use distinct local-chart titles', () => {
  const view = readResultComponent('ModelExplanation.vue')
  assert.ok(view.includes('当前样本预测类别线性得分贡献'), 'logistic title')
  assert.ok(view.includes('当前样本决策得分贡献'), 'SVM title')
  assert.ok(view.includes('类别线性得分（logit）'), 'logistic score wording')
  assert.ok(view.includes('决策得分'), 'SVM score wording')
})

// --- 7./8./9. KNN neighbours & votes ------------------------------------------------

test('KNN neighbour rows keep the recorded rank order', () => {
  const result = knnNeighborChartData(knnSample)
  assert.equal(result.available, true)
  assert.deepEqual(
    result.rows.map((row) => row.rank),
    [1, 2, 3],
  )
  assert.deepEqual(
    result.rows.map((row) => row.trainIndex),
    [12, 3, 7],
  )
  assert.equal(result.rows[0].name, '邻居 #1')

  const reordered = knnNeighborChartData({
    ...knnSample,
    neighbors: [knnSample.neighbors[1], knnSample.neighbors[0], knnSample.neighbors[2]],
  })
  assert.equal(reordered.available, false, 'non-ascending ranks are rejected')
})

test('KNN neighbour wording says 训练集内序号, never a raw dataset row', () => {
  const source = readResultComponent('KNNExplanation.vue')
  assert.ok(source.includes('训练集内序号'), 'neighbour tooltip wording')
  assert.ok(!source.includes('原始数据集行号'), 'misleading row wording')
  assert.ok(source.includes('邻居距离'))
  assert.ok(source.includes('投票权重'))
  assert.ok(source.includes('类别投票'))
})

test('KNN vote totals are displayed as recorded, never recomputed', () => {
  const result = knnVoteChartData(knnSample)
  assert.equal(result.available, true)
  assert.deepEqual(result.rows, [
    { classValue: 'a', weight: 12 },
    { classValue: 'b', weight: 1.111 },
  ])
  assert.equal(result.predicted, 'a')
  // The stored order is kept, even though class b has a smaller weight.
  assert.equal(result.rows[0].classValue, 'a')
})

// --- 10. class display mapping -------------------------------------------------------

test('class display names resolve through score_class_names with raw fallback', () => {
  const lookup = classDisplayLookup(['a', 'b'], ['类别甲', '类别乙'])
  assert.equal(classDisplayName('a', lookup), '类别甲')
  assert.equal(classDisplayName('b', lookup), '类别乙')

  const rawOnly = classDisplayLookup([0, 1], null)
  assert.equal(classDisplayName(0, rawOnly), '0')
})

// --- 11./12. GaussianNB likelihoods -----------------------------------------------------

const gnbSample = {
  sample_index: 1,
  y_true: 'x',
  y_pred: 'x',
  predicted_class: 'x',
  class_probabilities: [
    { class: 'x', probability: 0.8 },
    { class: 'y', probability: 0.2 },
  ],
  feature_values: [0.1, -0.2],
  feature_log_likelihoods: [-0.5, -1.5],
  log_prior: -0.7,
  joint_log_likelihood: -2.7,
}

test('GaussianNB likelihood rows align with feature names and keep signs', () => {
  const result = gaussianLikelihoodChartData({ sample: gnbSample, featureNames: ['f1', 'f2'] })
  assert.equal(result.available, true)
  assert.deepEqual(
    result.rows.map((row) => [row.feature, row.featureValue, row.logLikelihood]),
    [
      ['f1', 0.1, -0.5],
      ['f2', -0.2, -1.5],
    ],
  )
  assert.equal(result.predictedProbability, 0.8)
  assert.equal(result.logPrior, -0.7)

  const mismatched = gaussianLikelihoodChartData({ sample: gnbSample, featureNames: ['f1'] })
  assert.equal(mismatched.available, false)

  const brokenJoint = gaussianLikelihoodChartData({
    sample: { ...gnbSample, joint_log_likelihood: 5 },
    featureNames: ['f1', 'f2'],
  })
  assert.equal(brokenJoint.available, false)
})

test('GaussianNB likelihood is never called a probability contribution', () => {
  const source = readResultComponent('GaussianNBLocalLikelihoodChart.vue')
  assert.ok(source.includes('对数似然'))
  assert.ok(!source.includes('概率贡献'))
  assert.ok(source.includes('预测类别概率'), 'probability is shown as its own tag')
})

// --- 13./14. decision path --------------------------------------------------------------

const pathSample = {
  sample_index: 1,
  y_true: 150,
  y_pred: 152.37,
  path: [
    { step: 1, feature_index: 2, feature_name: 'bmi', value: 0.425, threshold: 0.731, direction: 'left' },
    { step: 2, feature_index: 8, feature_name: 'bp', value: 1.204, threshold: 0.552, direction: 'right' },
  ],
  leaf_prediction: 152.37,
}

test('decision path rows keep order, direction and leaf prediction', () => {
  const result = decisionPathRows(pathSample)
  assert.equal(result.available, true)
  assert.equal(result.steps.length, 2)
  assert.equal(result.steps[0].text, '第 1 步：bmi = 0.425 ≤ 0.731，进入左分支')
  assert.equal(result.steps[1].text, '第 2 步：bp = 1.204 > 0.552，进入右分支')
  assert.equal(result.leafText, '叶节点预测值：152.37')
  assert.equal(result.leafPrediction, 152.37)

  const wrongDirection = decisionPathRows({
    ...pathSample,
    path: [{ ...pathSample.path[0], direction: 'right' }],
  })
  assert.equal(wrongDirection.available, false, 'direction must match the comparison')
})

// --- 15./16./17. permutation importance ----------------------------------------------------

const mlpPayload = {
  schema_version: 1,
  kind: 'permutation_importance',
  feature_space: 'standardized',
  feature_names: ['a', 'b', 'c'],
  baseline_accuracy: 0.9333,
  importance_name: 'test_permutation_accuracy_drop',
  importances: [0.2, -0.05, 0],
}

const rfPayload = {
  schema_version: 1,
  kind: 'permutation_importance',
  feature_space: 'standardized',
  feature_names: ['a', 'b', 'c'],
  baseline_mse: 3200.5,
  importance_name: 'test_permutation_mse_increase',
  importances: [1500, -30.5, 0],
}

test('MLP permutation importance maps to the accuracy-drop labels', () => {
  const result = permutationImportanceData(mlpPayload)
  assert.equal(result.available, true)
  assert.equal(result.metric, 'accuracy')
  assert.equal(result.baselineLabel, '基准准确率')
  assert.equal(result.deltaLabel, '准确率变化')
  assert.equal(result.metricTag, '准确率下降')
  assert.equal(result.baseline, 0.9333)
})

test('RF/GBDT permutation importance maps to the MSE increase labels', () => {
  const result = permutationImportanceData(rfPayload)
  assert.equal(result.available, true)
  assert.equal(result.metric, 'mse')
  assert.equal(result.baselineLabel, '基准 MSE')
  assert.equal(result.deltaLabel, 'MSE 增量')
  assert.equal(result.metricTag, 'MSE 增量')
  assert.equal(result.baseline, 3200.5)
})

test('negative permutation importances are kept and sorted stably', () => {
  const result = permutationImportanceData(rfPayload)
  assert.deepEqual(
    result.rows.map((row) => row.importance),
    [1500, 0, -30.5],
  )
  assert.equal(result.rows[2].direction, 'negative')

  const equal = permutationImportanceData({
    ...rfPayload,
    importances: [1, 1, 0.5],
  })
  assert.deepEqual(
    equal.rows.map((row) => row.feature),
    ['a', 'b', 'c'],
    'equal importances keep the original feature order',
  )
})

// --- 18. sorting never mutates the payload --------------------------------------------------

test('permutation sorting never modifies the input payload', () => {
  const raw = JSON.parse(JSON.stringify(rfPayload))
  const snapshot = JSON.parse(JSON.stringify(raw))
  permutationImportanceData(raw)
  assert.deepEqual(raw, snapshot)
})

// --- 19. legacy experiments without explanation_data -----------------------------------------

test('missing or invalid explanation data falls back without local cards', () => {
  assert.equal(normalizeExplanationData(undefined).available, false)
  assert.equal(normalizeExplanationData(null).available, false)
  assert.equal(explanationSamples(normalizeExplanationData(null)).length, 0)

  const view = readResultComponent('ModelExplanation.vue')
  // Every local component is guarded by recorded data, so legacy experiments
  // keep only their existing global charts.
  assert.ok(/<LocalFeatureContributionChart[\s\S]*?v-if="hasSamples"/.test(view))
  assert.ok(/<GaussianNBLocalLikelihoodChart[\s\S]*?v-if="hasSamples"/.test(view))
  assert.ok(view.includes("modelName === 'decision_tree_regressor' && hasSamples"))
  assert.ok(!view.includes('历史实验'), 'no legacy-experiment prose is added')
})

// --- 20./21./22. wording scans ---------------------------------------------------------------

test('forbidden explanatory phrases stay removed from production code', () => {
  const banned = [
    '解释说明',
    '重要性说明',
    '系数说明',
    '为什么没有损失曲线',
    '不代表因果关系',
    '不代表特征选择结论',
    '机制说明不代表',
    '本次训练记录',
    '概率贡献',
    '越大越好',
    '最优',
  ]
  const failures = []
  for (const source of productionSources()) {
    for (const phrase of banned) {
      if (source.includes(phrase)) failures.push(phrase)
    }
  }
  assert.deepEqual(failures, [], `forbidden wording found: ${failures.join(', ')}`)
})

test('no fixed English titles leak into the new components', () => {
  const banned = [
    'Local Contribution',
    'Neighbor Distance',
    'Vote Weight',
    'Decision Path',
    'Permutation Importance',
    'Baseline Accuracy',
    'Baseline MSE',
    'Log Likelihood',
  ]
  const failures = []
  for (const name of NEW_COMPONENTS) {
    const source = readResultComponent(name)
    for (const phrase of banned) {
      if (source.includes(phrase)) failures.push(`${name}: ${phrase}`)
    }
  }
  assert.deepEqual(failures, [], `untranslated wording found: ${failures.join(', ')}`)
})

test('all agreed Chinese labels are present in the new components', () => {
  const expectations = {
    'ModelExplanation.vue': ['当前样本特征贡献', '当前样本预测类别线性得分贡献', '当前样本决策得分贡献'],
    'LocalFeatureContributionChart.vue': ['截距', '贡献合计', '标准化后特征值', '贡献值'],
    'PermutationImportanceChart.vue': ['测试集置换重要性'],
    'GaussianNBLocalLikelihoodChart.vue': ['当前样本特征对数似然', '对数似然'],
    'DecisionPathExplanation.vue': ['当前样本决策路径', '叶节点预测值'],
    'KNNExplanation.vue': ['邻居距离', '类别投票', '投票权重', '训练集内序号'],
    'SampleExplanationSelector.vue': ['解释对象', '测试样本'],
  }
  for (const [name, phrases] of Object.entries(expectations)) {
    const source = readResultComponent(name)
    for (const phrase of phrases) {
      assert.ok(source.includes(phrase), `${name} must contain ${phrase}`)
    }
  }

  const utils = readSource('utils/modelExplanation.js')
  for (const phrase of ['左分支', '右分支', '基准准确率', '基准 MSE', '准确率变化', '准确率下降', 'MSE 增量', '叶节点预测值']) {
    assert.ok(utils.includes(phrase), `modelExplanation.js must contain ${phrase}`)
  }
})

// --- 23./24./26. layout & responsive rules ----------------------------------------------------

test('comparable rows use the equal 1fr/1fr grid with stretched children', () => {
  const gridded = ['ModelExplanation.vue', 'KNNExplanation.vue', 'GaussianNBExplanation.vue']
  for (const name of gridded) {
    const source = readResultComponent(name)
    assert.ok(source.includes('repeat(2, minmax(0, 1fr))'), `${name} equal columns`)
    assert.ok(source.includes('align-items: stretch'), `${name} stretches rows`)
    assert.ok(source.includes('min-width: 0'), `${name} prevents grid overflow`)
  }
  for (const source of resultComponentSources()) {
    assert.ok(!source.includes('2fr'), 'found a 2fr track in a result component')
  }
})

test('responsive breakpoints keep single-column and compact paddings', () => {
  for (const name of ['ModelExplanation.vue', 'KNNExplanation.vue', 'GaussianNBExplanation.vue']) {
    const source = readResultComponent(name)
    assert.ok(source.includes('max-width: 1100px'), `${name} single column ≤1100px`)
  }
  for (const name of NEW_COMPONENTS) {
    const source = readResultComponent(name)
    assert.ok(source.includes('max-width: 720px'), `${name} compact padding ≤720px`)
  }
})

test('wide-feature charts keep the inside + slider dataZoom', () => {
  for (const name of [
    'LocalFeatureContributionChart.vue',
    'GaussianNBLocalLikelihoodChart.vue',
    'PermutationImportanceChart.vue',
    'MulticlassCoefficientHeatmap.vue',
  ]) {
    const source = readResultComponent(name)
    assert.ok(source.includes("type: 'inside'"), `${name} inside zoom`)
    assert.ok(source.includes("type: 'slider'"), `${name} slider zoom`)
  }
})

// --- 25. no new backend requests ---------------------------------------------------------------

test('the explanation tab never issues its own network requests', () => {
  for (const name of [...NEW_COMPONENTS, 'MLPArchitecture.vue']) {
    const source = readResultComponent(name)
    for (const marker of ['fetch(', 'axios', '/api/', 'XMLHttpRequest']) {
      assert.ok(!source.includes(marker), `${name} must not call ${marker}`)
    }
  }
})

// --- chart height & lifecycle ------------------------------------------------------------------

test('all new chart areas are 350px and reuse the shared chart lifecycle', () => {
  const charted = [
    'LocalFeatureContributionChart.vue',
    'PermutationImportanceChart.vue',
    'GaussianNBLocalLikelihoodChart.vue',
    'KNNExplanation.vue',
  ]
  for (const name of charted) {
    const source = readResultComponent(name)
    assert.ok(source.includes('height: 350px'), `${name} chart height`)
    assert.ok(source.includes("from './useChart'"), `${name} reuses useChart`)
    assert.ok(!source.includes("window.addEventListener('resize'"), `${name} leaves resize to useChart`)
  }
  const path = readResultComponent('DecisionPathExplanation.vue')
  assert.ok(path.includes('height: 350px'), 'decision path body height')
  assert.ok(path.includes('overflow-y: auto'), 'decision path scrolls internally')
})
