/**
 * Phase 7B1 — result page localisation contract tests (Node built-in runner).
 *
 * Scans production sources as text to guarantee that user-visible English was
 * translated, that agreed professional abbreviations survived, that raw
 * dataset feature/class names are never "translated", and that the wording
 * removed in Phase 7A stays removed.
 *
 * Run with:
 *   node --test tests/frontend/result_localization.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const SRC_ROOT = fileURLToPath(new URL('../../frontend/src/', import.meta.url))
const RESULT_DIR = `${SRC_ROOT}components/result/`

const readSource = (relativePath) => readFileSync(`${SRC_ROOT}${relativePath}`, 'utf8')

const resultComponentSources = () =>
  readdirSync(RESULT_DIR)
    .filter((file) => file.endsWith('.vue'))
    .map((file) => readFileSync(`${RESULT_DIR}${file}`, 'utf8'))

const productionSources = () => [
  readSource('views/ResultView.vue'),
  ...resultComponentSources(),
  readSource('utils/training.js'),
  readSource('utils/resultLabels.js'),
  readSource('utils/modelExplanation.js'),
  readSource('utils/resultDiagnostics.js'),
]

// --- 1. translated visible English stays gone ---------------------------------

test('translated visible English labels no longer appear in production code', () => {
  // Quoted chart series names, tooltip prefixes and the page eyebrow.
  const banned = [
    "name: 'Precision'",
    "name: 'Recall'",
    'Precision：',
    'Recall：',
    'Precision / Recall',
    'MODEL INSIGHTS',
    '>Top ',
  ]
  const failures = []
  for (const source of productionSources()) {
    for (const phrase of banned) {
      if (source.includes(phrase)) failures.push(phrase)
    }
  }
  assert.deepEqual(failures, [], `untranslated wording still present: ${failures.join(', ')}`)
})

test('the per-class metrics chart uses the Chinese series names', () => {
  const source = readFileSync(`${RESULT_DIR}PerClassMetricsChart.vue`, 'utf8')
  assert.ok(source.includes("name: '精确率'"))
  assert.ok(source.includes("name: '召回率'"))
  assert.ok(source.includes('精确率 / 召回率 / F1'))
})

// --- 2. professional abbreviations survive --------------------------------------

test('common professional abbreviations are kept', () => {
  const resultView = readSource('views/ResultView.vue')
  for (const abbr of ['MAE', 'MSE', 'RMSE', 'R²', 'KNN', 'SVM', 'GBDT']) {
    assert.ok(resultView.includes(abbr), `ResultView must keep ${abbr}`)
  }

  const mlpArchitecture = readFileSync(`${RESULT_DIR}MLPArchitecture.vue`, 'utf8')
  assert.ok(mlpArchitecture.includes('MLP'), 'MLPArchitecture must keep MLP')

  const rocChart = readFileSync(`${RESULT_DIR}RocCurveChart.vue`, 'utf8')
  assert.ok(rocChart.includes('ROC'))
  assert.ok(rocChart.includes('AUC'))

  const labels = readSource('utils/resultLabels.js')
  assert.ok(labels.includes("accuracy: 'Accuracy'"))
  assert.ok(labels.includes("f1_weighted: 'F1'"))
})

// --- 3. raw dataset names are never translated ----------------------------------

test('raw dataset feature and class names are not translated', () => {
  const bannedTranslations = [
    '山鸢尾',
    '变色鸢尾',
    '维吉尼亚鸢尾',
    '恶性（malignant',
    '良性（benign',
  ]
  for (const source of productionSources()) {
    for (const phrase of bannedTranslations) {
      assert.ok(!source.includes(phrase), `raw name translation leaked: ${phrase}`)
    }
  }

  // The log helpers must not carry any dataset-specific wording at all.
  const labels = readSource('utils/resultLabels.js')
  for (const raw of ['setosa', 'versicolor', 'malignant', 'benign', 'pixel_']) {
    assert.ok(!labels.includes(raw), `resultLabels must stay dataset-agnostic: ${raw}`)
  }
})

// --- 4. Phase 7A removals stay removed -------------------------------------------

test('Phase 7A removed wording does not reappear', () => {
  const removed = [
    '解释说明',
    '重要性说明',
    '系数说明',
    '训练过程说明',
    '结果解读',
    '为什么没有损失曲线',
    '不代表因果关系',
    '不代表特征选择结论',
    '机制说明不代表',
    '本次训练记录',
  ]
  const failures = []
  for (const source of productionSources()) {
    for (const phrase of removed) {
      if (source.includes(phrase)) failures.push(phrase)
    }
  }
  assert.deepEqual(failures, [], `removed wording reappeared: ${failures.join(', ')}`)
})

// --- 5. log panel wiring -----------------------------------------------------------

test('the training log panel is mounted and self-hides without a valid log', () => {
  const process = readFileSync(`${RESULT_DIR}TrainingProcess.vue`, 'utf8')
  assert.ok(process.includes('TrainingProcessLog'), 'TrainingProcess must mount the log panel')
  assert.ok(process.includes(':process-log="info.process_log"'))

  const panel = readFileSync(`${RESULT_DIR}TrainingProcessLog.vue`, 'utf8')
  assert.ok(/v-if="hasLines"/.test(panel), 'the card must hide entirely when there are no lines')
  assert.ok(panel.includes('训练日志'), 'card title')
  assert.ok(panel.includes('overflow-y: auto'), 'scrollable log body')
  assert.ok(panel.includes('360px'), 'capped log height')
  assert.ok(panel.includes('monospace'), 'monospace log font')
})
