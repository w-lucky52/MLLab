/**
 * Module-level registry of exportable ECharts chart providers (Phase 7C).
 *
 * Each chart component registers its own ``buildOption`` when it mounts and
 * unregisters on unmount. Because every ``el-tab-pane`` is rendered up front
 * (Element Plus renders non-lazy panes immediately), a chart is registered
 * even when the user never visits its tab. ``ResultView`` owns the lifecycle:
 * it clears the registry whenever the route changes so a previous experiment's
 * providers can never leak into the next one.
 *
 * A provider has the shape:
 *   { id, section, title, order, available: () => boolean, buildOption: () => object }
 */

const providers = new Map()

export function registerChartProvider(provider) {
  if (!provider || typeof provider.id !== 'string' || provider.id === '') return
  providers.set(provider.id, provider)
}

export function unregisterChartProvider(id) {
  providers.delete(id)
}

export function clearChartProviders() {
  providers.clear()
}

export function getChartProviders() {
  return Array.from(providers.values())
}
