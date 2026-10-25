import { onBeforeUnmount, onMounted } from 'vue'
import { registerChartProvider, unregisterChartProvider } from '@/utils/exportRegistry'

/**
 * Register an ECharts chart for the "export as ZIP" flow.
 *
 * ``available`` must be a getter (e.g. ``() => hasData.value``) so the export
 * pipeline can re-evaluate it against the latest data — including a newly
 * selected test sample — without the component having to re-register.
 * ``buildOption`` is the component's own pure option builder, whose closures
 * already track the latest reactive data.
 *
 * @param {{ id: string, section: string, title: string, order: number,
 *           available: () => boolean, buildOption: () => object }} config
 */
export function useExportableChart({ id, section, title, order, available, buildOption }) {
  onMounted(() => {
    registerChartProvider({ id, section, title, order, available, buildOption })
  })

  onBeforeUnmount(() => {
    unregisterChartProvider(id)
  })
}
