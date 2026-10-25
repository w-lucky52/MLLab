/**
 * Pure helpers for the "export all charts as a ZIP" flow (Phase 7C).
 *
 * Everything here is free of DOM, ECharts and JSZip so it can be exercised
 * directly by the Node test runner. The helpers build the ZIP file name,
 * sanitise Windows-hostile characters, order providers deterministically,
 * turn providers into per-folder PNG entries and split PNG data URLs.
 */

// Fixed section order used when writing the ZIP folders.
export const SECTION_ORDER = ['分类表现', '回归表现', '训练过程', '模型洞察']

// Characters that are illegal in Windows file names.
const ILLEGAL_FILENAME_CHARS = /[\\/:*?"<>|]/g

/**
 * Remove characters that are illegal in Windows file names. An empty result
 * falls back to "图表" so we never emit an empty file name.
 */
export function sanitizeFilename(name) {
  const text = String(name ?? '')
    .replace(ILLEGAL_FILENAME_CHARS, '_')
    .trim()
  return text || '图表'
}

/** Build the ZIP file name, e.g. ``实验80_iris_logistic_regression_图表.zip``. */
export function buildZipFilename({ id, datasetName, modelName }) {
  const parts = [id, datasetName, modelName].map((value) => String(value ?? '').trim())
  return `实验${parts.join('_')}_图表.zip`
}

/** Keep only providers that report themselves as currently available. */
export function filterAvailableProviders(providers) {
  if (!Array.isArray(providers)) return []
  return providers.filter(
    (provider) =>
      provider &&
      typeof provider.available === 'function' &&
      provider.available() === true,
  )
}

/**
 * Order providers by section (fixed order above, unknown sections last) and
 * then by ``order``. Array.prototype.sort is stable, so providers sharing a
 * section and order keep their original relative order.
 */
export function sortProviders(providers) {
  const sectionIndex = (section) => {
    const index = SECTION_ORDER.indexOf(section)
    return index === -1 ? SECTION_ORDER.length : index
  }
  return [...providers].sort((left, right) => {
    const sectionDiff = sectionIndex(left.section) - sectionIndex(right.section)
    if (sectionDiff !== 0) return sectionDiff
    return (left.order ?? 0) - (right.order ?? 0)
  })
}

/**
 * Turn sorted providers into ZIP entries. Each entry gets a two-digit
 * per-section sequence number so duplicate titles cannot overwrite each
 * other, and the file name keeps its Chinese title (sanitised).
 */
export function buildZipEntries(providers) {
  const counts = new Map()
  return providers.map((provider) => {
    const section = sanitizeFilename(provider.section)
    const count = (counts.get(provider.section) ?? 0) + 1
    counts.set(provider.section, count)
    const number = String(count).padStart(2, '0')
    const title = sanitizeFilename(provider.title)
    return {
      section: provider.section,
      folder: section,
      filename: `${number}_${title}.png`,
      number,
      title: provider.title,
      provider,
    }
  })
}

/** True when the value is a PNG data URL with the expected prefix. */
export function isPngDataUrl(dataUrl) {
  return typeof dataUrl === 'string' && dataUrl.startsWith('data:image/png;base64,')
}

/** Strip the ``data:image/png;base64,`` prefix; non-PNG input yields ''. */
export function dataUrlToBase64(dataUrl) {
  if (!isPngDataUrl(dataUrl)) return ''
  return dataUrl.slice('data:image/png;base64,'.length)
}
