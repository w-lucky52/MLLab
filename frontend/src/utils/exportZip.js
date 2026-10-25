import JSZip from 'jszip'
import { dataUrlToBase64 } from './exportChartUtils.js'

/**
 * Populate a JSZip archive with one PNG per entry, nested under a root folder
 * and per-section sub-folders. Callers can either inspect the zip in tests or
 * ``generateAsync`` it themselves.
 *
 * @param {JSZip} zip target archive
 * @param {string} rootName root folder name inside the archive
 * @param {Array<{ folder: string, filename: string, dataUrl: string }>} entries
 * @returns {JSZip} the same archive (for chaining)
 */
export function populateZip(zip, rootName, entries) {
  const root = zip.folder(rootName)
  const folders = new Map()

  for (const entry of entries) {
    if (!folders.has(entry.folder)) {
      folders.set(entry.folder, root.folder(entry.folder))
    }
    const folder = folders.get(entry.folder)
    const base64 = dataUrlToBase64(entry.dataUrl)
    if (base64) {
      folder.file(entry.filename, base64, { base64: true })
    }
  }

  return zip
}

/**
 * Build and generate the export ZIP as a Blob (used by the browser download).
 */
export async function generateZipBlob(rootName, entries) {
  const zip = populateZip(new JSZip(), rootName, entries)
  return zip.generateAsync({ type: 'blob' })
}
