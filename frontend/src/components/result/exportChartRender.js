import * as echarts from 'echarts'
import { isPngDataUrl } from '@/utils/exportChartUtils'

/**
 * Render a chart's option off-screen and return a PNG data URL.
 *
 * The container is fixed outside the viewport (never ``display:none``), sized
 * to a fixed 820×460 so every chart has a deterministic canvas. Rendering is
 * synchronous because animation is disabled; the instance and its DOM node are
 * disposed immediately afterwards so nothing leaks.
 *
 * @param {() => object} buildOption provider option builder
 * @returns {string} PNG data URL (empty string on failure)
 */
export function renderChartToPng(buildOption) {
  let element = null
  let chart = null
  try {
    const option = buildOption()
    if (!option) return ''

    element = document.createElement('div')
    element.style.cssText =
      'position:fixed;left:-10000px;top:-10000px;width:820px;height:460px;'
    document.body.appendChild(element)

    chart = echarts.init(element)
    chart.setOption({ ...option, animation: false }, true)

    const dataUrl = chart.getDataURL({
      type: 'png',
      pixelRatio: 2,
      backgroundColor: '#ffffff',
    })
    return isPngDataUrl(dataUrl) ? dataUrl : ''
  } catch {
    return ''
  } finally {
    if (chart) chart.dispose()
    if (element && element.parentNode) element.parentNode.removeChild(element)
  }
}
