/**
 * Phase 7C — export-all-charts-as-ZIP contract tests (Node built-in runner).
 *
 * Pure helpers (file naming, ordering, folder layout, data-URL handling and
 * the JSZip population) are exercised directly; production sources are also
 * scanned as text to pin down the export entry point, the single dependency
 * and the chart-component registration coverage.
 *
 * Run with:
 *   node --test tests/frontend/result_export.test.mjs
 */

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import JSZip from '../../frontend/node_modules/jszip/lib/index.js'

import {
  SECTION_ORDER,
  sanitizeFilename,
  buildZipFilename,
  filterAvailableProviders,
  sortProviders,
  buildZipEntries,
  isPngDataUrl,
  dataUrlToBase64,
} from '../../frontend/src/utils/exportChartUtils.js'
import { populateZip } from '../../frontend/src/utils/exportZip.js'
import {
  registerChartProvider,
  unregisterChartProvider,
  clearChartProviders,
  getChartProviders,
} from '../../frontend/src/utils/exportRegistry.js'

const SRC_ROOT = fileURLToPath(new URL('../../frontend/src/', import.meta.url))
const RESULT_DIR = `${SRC_ROOT}components/result/`

const readSource = (relativePath) => readFileSync(`${SRC_ROOT}${relativePath}`, 'utf8')
const readResultComponent = (name) => readSource(`components/result/${name}`)

const PNG = 'data:image/png;base64,QUJD'

// --- 1./2./3. file naming ----------------------------------------------------

test('buildZipFilename composes the agreed Chinese ZIP name', () => {
  assert.equal(
    buildZipFilename({ id: 80, datasetName: 'iris', modelName: 'logistic_regression' }),
    '实验80_iris_logistic_regression_图表.zip',
  )
})

test('sanitizeFilename strips Windows-illegal characters', () => {
  assert.equal(sanitizeFilename('a/b\\c'), 'a_b_c')
  assert.equal(sanitizeFilename('a:b*c?d"e<f>g|h'), 'a_b_c_d_e_f_g_h')
  assert.equal(sanitizeFilename(''), '图表')
  assert.equal(sanitizeFilename('   '), '图表')
})

test('sanitizeFilename keeps Chinese titles intact', () => {
  assert.equal(sanitizeFilename('混淆矩阵'), '混淆矩阵')
  assert.equal(sanitizeFilename('测试集置换重要性'), '测试集置换重要性')
})

// --- 4./5./6. ordering -------------------------------------------------------

test('sortProviders orders sections by the fixed SECTION_ORDER', () => {
  const providers = [
    { section: '模型洞察', order: 1 },
    { section: '分类表现', order: 1 },
    { section: '训练过程', order: 1 },
    { section: '回归表现', order: 1 },
  ]
  assert.deepEqual(
    sortProviders(providers).map((p) => p.section),
    ['分类表现', '回归表现', '训练过程', '模型洞察'],
  )
})

test('sortProviders orders by provider order within a section', () => {
  const providers = [
    { section: '分类表现', order: 3 },
    { section: '分类表现', order: 1 },
    { section: '分类表现', order: 2 },
  ]
  assert.deepEqual(
    sortProviders(providers).map((p) => p.order),
    [1, 2, 3],
  )
})

test('sortProviders is stable for equal order values', () => {
  const providers = [
    { section: '分类表现', order: 1, title: 'A' },
    { section: '分类表现', order: 1, title: 'B' },
    { section: '分类表现', order: 1, title: 'C' },
  ]
  assert.deepEqual(
    sortProviders(providers).map((p) => p.title),
    ['A', 'B', 'C'],
  )
})

// --- 7./8./19. ZIP entries ---------------------------------------------------

test('buildZipEntries uses two-digit sequence numbers', () => {
  const providers = [
    { section: '分类表现', title: '混淆矩阵', order: 1 },
    { section: '分类表现', title: 'ROC 曲线', order: 2 },
  ]
  const entries = buildZipEntries(providers)
  assert.equal(entries[0].filename, '01_混淆矩阵.png')
  assert.equal(entries[1].filename, '02_ROC 曲线.png')
})

test('duplicate titles cannot overwrite each other', () => {
  const providers = [
    { section: '分类表现', title: '同名列', order: 1 },
    { section: '分类表现', title: '同名列', order: 2 },
  ]
  const entries = buildZipEntries(providers)
  assert.notEqual(entries[0].filename, entries[1].filename)
})

test('buildZipEntries never emits undefined, null or NaN', () => {
  const providers = [
    { section: '分类表现', title: '混淆矩阵', order: 1 },
    { section: '模型洞察', title: '特征系数', order: 1 },
  ]
  for (const entry of buildZipEntries(providers)) {
    for (const value of [entry.section, entry.folder, entry.filename, entry.number, entry.title]) {
      assert.equal(typeof value, 'string')
      assert.ok(value.length > 0)
      assert.ok(!/undefined|null|NaN/.test(value))
    }
    assert.ok(entry.filename.endsWith('.png'))
  }
})

// --- 9./12./13. provider filtering -------------------------------------------

test('filterAvailableProviders drops invalid and unavailable providers', () => {
  const providers = [
    null,
    { section: '分类表现', title: 'x', order: 1 },
    { section: '分类表现', title: 'x', order: 1, available: () => false },
    { section: '分类表现', title: 'x', order: 1, available: () => true },
  ]
  assert.equal(filterAvailableProviders(providers).length, 1)
})

test('zero charts is handled safely by every pure helper', () => {
  assert.deepEqual(filterAvailableProviders([]), [])
  assert.deepEqual(sortProviders([]), [])
  assert.deepEqual(buildZipEntries([]), [])
})

test('legacy experiments export only their available charts', () => {
  const providers = [
    { section: '分类表现', title: '混淆矩阵', order: 1, available: () => true },
    { section: '分类表现', title: 'ROC 曲线', order: 2, available: () => false },
  ]
  const available = filterAvailableProviders(providers)
  assert.equal(available.length, 1)
  assert.equal(available[0].title, '混淆矩阵')
})

// --- 10./11. data URL handling ----------------------------------------------

test('dataUrlToBase64 strips the PNG prefix', () => {
  assert.equal(dataUrlToBase64('data:image/png;base64,QUJD'), 'QUJD')
})

test('non-PNG data URLs are rejected', () => {
  assert.equal(isPngDataUrl('data:image/jpeg;base64,QUJD'), false)
  assert.equal(isPngDataUrl('data:image/png;base64,QUJD'), true)
  assert.equal(dataUrlToBase64('data:image/jpeg;base64,QUJD'), '')
  assert.equal(dataUrlToBase64(''), '')
})

// --- 14./15./16./17. folder structure ---------------------------------------

async function buildTestZip(rootName, entries) {
  const zip = populateZip(new JSZip(), rootName, entries)
  const buffer = await zip.generateAsync({ type: 'nodebuffer' })
  return JSZip.loadAsync(buffer)
}

function zipEntries(providers) {
  return buildZipEntries(sortProviders(providers)).map((entry) => ({ ...entry, dataUrl: PNG }))
}

test('classification experiment writes the 分类表现 folder', async () => {
  const entries = zipEntries([
    { section: '分类表现', title: '每类别分类指标', order: 1 },
    { section: '分类表现', title: '混淆矩阵', order: 2 },
    { section: '分类表现', title: 'ROC 曲线', order: 3 },
    { section: '分类表现', title: '预测置信度分布', order: 4 },
  ])
  const zip = await buildTestZip('实验_root', entries)
  const folder = zip.folder('实验_root').folder('分类表现')
  assert.ok(folder.file('01_每类别分类指标.png'))
  assert.ok(folder.file('02_混淆矩阵.png'))
  assert.ok(folder.file('03_ROC 曲线.png'))
  assert.ok(folder.file('04_预测置信度分布.png'))
})

test('regression experiment writes the 回归表现 folder', async () => {
  const entries = zipEntries([
    { section: '回归表现', title: '真实值与预测值', order: 1 },
    { section: '回归表现', title: '预测值与残差', order: 2 },
    { section: '回归表现', title: '残差分布', order: 3 },
    { section: '回归表现', title: '绝对误差最大的样本', order: 4 },
  ])
  const zip = await buildTestZip('实验_root', entries)
  const folder = zip.folder('实验_root').folder('回归表现')
  assert.ok(folder.file('01_真实值与预测值.png'))
  assert.ok(folder.file('04_绝对误差最大的样本.png'))
})

test('训练过程 and 模型洞察 folders keep their agreed position', async () => {
  const entries = zipEntries([
    { section: '训练过程', title: '损失曲线', order: 1 },
    { section: '模型洞察', title: '特征系数', order: 1 },
    { section: '模型洞察', title: '类别系数热力图', order: 2 },
  ])
  const zip = await buildTestZip('实验_root', entries)
  assert.ok(zip.folder('实验_root').folder('训练过程').file('01_损失曲线.png'))
  const insight = zip.folder('实验_root').folder('模型洞察')
  assert.ok(insight.file('01_特征系数.png'))
  assert.ok(insight.file('02_类别系数热力图.png'))
})

// --- 18. input immutability --------------------------------------------------

test('sortProviders and buildZipEntries never mutate their input', () => {
  const providers = Object.freeze([
    Object.freeze({ section: '模型洞察', title: '特征系数', order: 1 }),
    Object.freeze({ section: '分类表现', title: '混淆矩阵', order: 1 }),
  ])
  const snapshot = providers.map((p) => ({ ...p }))
  sortProviders(providers)
  buildZipEntries(providers)
  assert.deepEqual(providers, snapshot)
})

// --- 20./21./22. entry point scanning ---------------------------------------

test('ResultView keeps a single export entry at the bottom footer', () => {
  const source = readSource('views/ResultView.vue')
  assert.ok(source.includes('导出结果'))
  assert.equal(source.indexOf('导出结果'), source.lastIndexOf('导出结果'))
  // The heading section must not grow a second export button.
  const heading = source.slice(source.indexOf('page-heading'), source.indexOf('experiment-summary'))
  assert.ok(!heading.includes('导出结果'))
})

test('exportResult never issues an experiment API request', () => {
  const source = readSource('views/ResultView.vue')
  const start = source.indexOf('const exportResult =')
  const end = source.indexOf('\nwatch(exportDialogVisible', start)
  const body = source.slice(start, end)
  assert.ok(!body.includes('fetch('))
  assert.ok(body.includes('renderChartToPng'))
})

// --- 23. provider lifecycle --------------------------------------------------

test('providers register, unregister and clear correctly', () => {
  clearChartProviders()
  registerChartProvider({ id: 'a', section: '分类表现', title: 'x', order: 1 })
  registerChartProvider({ id: 'b', section: '分类表现', title: 'y', order: 2 })
  assert.equal(getChartProviders().length, 2)

  unregisterChartProvider('a')
  assert.equal(getChartProviders().length, 1)

  clearChartProviders()
  assert.equal(getChartProviders().length, 0)
})

// --- 24./25. download mechanics ----------------------------------------------

test('download releases the blob URL and clicks the anchor exactly once', () => {
  const source = readResultComponent('ExportResultDialog.vue')
  assert.ok(source.includes('URL.createObjectURL'))
  assert.ok(source.includes('URL.revokeObjectURL'))
  assert.ok(source.includes('anchor.click()'))
  assert.equal(source.split('anchor.click()').length - 1, 1)
  assert.ok(source.includes("if (downloading.value) return"))
})

// --- 26./27. dependency constraints ------------------------------------------

test('package.json adds only jszip and no other dependency', () => {
  const pkg = JSON.parse(readFileSync(`${SRC_ROOT}../package.json`, 'utf8'))
  const deps = Object.keys(pkg.dependencies).sort()
  assert.deepEqual(deps, [
    '@element-plus/icons-vue',
    'echarts',
    'element-plus',
    'jszip',
    'vue',
    'vue-router',
  ])
})

test('production code never references html2canvas or file-saver', () => {
  const resultSources = readdirSync(RESULT_DIR)
    .filter((file) => file.endsWith('.vue'))
    .map((file) => readFileSync(`${RESULT_DIR}${file}`, 'utf8'))
  const sources = [
    readSource('views/ResultView.vue'),
    ...resultSources,
    readSource('utils/exportChartUtils.js'),
    readSource('utils/exportZip.js'),
    readSource('utils/exportRegistry.js'),
  ]
  for (const source of sources) {
    assert.ok(!source.includes('html2canvas'))
    assert.ok(!source.includes('file-saver'))
  }
})

// --- 28. registration coverage ----------------------------------------------

const EXPORTABLE_COMPONENTS = [
  'PerClassMetricsChart.vue',
  'ConfusionMatrixChart.vue',
  'RocCurveChart.vue',
  'PredictionConfidenceChart.vue',
  'ActualPredictedChart.vue',
  'ResidualScatterChart.vue',
  'ResidualHistogramChart.vue',
  'TopAbsoluteErrorsChart.vue',
  'TrainingLossChart.vue',
  'CoefficientChart.vue',
  'MulticlassCoefficientHeatmap.vue',
  'FeatureImportanceChart.vue',
  'GaussianNBExplanation.vue',
  'KNNExplanation.vue',
  'LocalFeatureContributionChart.vue',
  'GaussianNBLocalLikelihoodChart.vue',
  'PermutationImportanceChart.vue',
]

test('every ECharts chart component registers its buildOption', () => {
  for (const name of EXPORTABLE_COMPONENTS) {
    const source = readResultComponent(name)
    assert.ok(source.includes('useExportableChart'), `${name} must register for export`)
  }
})

test('non-chart modules are never registered for export', () => {
  for (const name of [
    'AlgorithmTrainingFlow.vue',
    'DecisionPathExplanation.vue',
    'MLPArchitecture.vue',
    'SampleExplanationSelector.vue',
    'TrainingProcessLog.vue',
  ]) {
    const source = readResultComponent(name)
    assert.ok(!source.includes('useExportableChart'), `${name} must not register for export`)
  }
})
