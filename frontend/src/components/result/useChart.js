import { nextTick, onBeforeUnmount, onMounted, watch } from 'vue'
import * as echarts from 'echarts'

/**
 * Own one ECharts instance for a component.
 *
 * The instance is created only when the container is visible (``isActive``)
 * and after ``nextTick``, so hidden tab panes never produce zero-size charts.
 * Switching back to a tab re-runs the option and resizes the instance instead
 * of creating a second one. The instance is disposed when the component
 * unmounts, and window resizes are forwarded to it.
 *
 * @param {import('vue').Ref<HTMLElement | null>} elRef chart container ref
 * @param {() => object} getOption builds the ECharts option from current data
 * @param {() => boolean} isActive whether the container is visible
 */
export function useChart(elRef, getOption, isActive) {
  let instance = null

  const render = () => {
    if (!instance) return
    instance.setOption(getOption(), true)
  }

  const resize = () => {
    instance?.resize()
  }

  const mount = async () => {
    if (!isActive()) return
    await nextTick()
    if (!isActive()) return

    if (!elRef.value) {
      // The container was removed (e.g. switched to an empty state).
      dispose()
      return
    }

    // The container may have been replaced by a new DOM node; an instance
    // bound to the old node would leak and never display again.
    if (instance && instance.getDom() !== elRef.value) {
      dispose()
    }
    if (!instance) {
      instance = echarts.init(elRef.value)
    }
    render()
    resize()
  }

  const dispose = () => {
    instance?.dispose()
    instance = null
  }

  onMounted(() => {
    window.addEventListener('resize', resize)
    mount()
  })

  watch(isActive, (active) => {
    if (active) mount()
  })

  onBeforeUnmount(() => {
    window.removeEventListener('resize', resize)
    dispose()
  })

  return { mount, render, resize }
}
