/**
 * Centralized display-label mapping for classification results.
 *
 * ``score_classes`` are the real class values used by scoring, ROC and the
 * confusion matrix. ``score_class_names`` is an optional, display-only list
 * aligned with those classes (same order). These helpers never mutate or
 * re-order the underlying data; they only produce labels for the UI.
 */

const isNonEmptyString = (value) => typeof value === 'string' && value.trim().length > 0

/**
 * Return one display label per score class.
 *
 * Friendly names are used only when both arrays have the same non-zero
 * length and every name is a non-empty string. Otherwise every class falls
 * back to ``String(class)`` (e.g. the raw 0/1/2 of older experiments).
 */
export function resolveClassDisplayNames(scoreClasses, scoreClassNames) {
  const classes = Array.isArray(scoreClasses) ? scoreClasses : []
  const names = Array.isArray(scoreClassNames) ? scoreClassNames : []
  const usable = classes.length > 0 && names.length === classes.length && names.every(isNonEmptyString)
  return classes.map((value, index) => (usable ? names[index].trim() : String(value)))
}

/**
 * Build a ``String(classValue) -> displayName`` map in the original order.
 *
 * Matching by value (not position) keeps row/column order intact even when a
 * component receives classes in a different order. Duplicate values keep the
 * first mapping.
 */
export function classDisplayLookup(scoreClasses, scoreClassNames) {
  const classes = Array.isArray(scoreClasses) ? scoreClasses : []
  const names = resolveClassDisplayNames(classes, scoreClassNames)
  const lookup = new Map()
  classes.forEach((value, index) => {
    const key = String(value)
    if (!lookup.has(key)) lookup.set(key, names[index])
  })
  return lookup
}

/** Resolve one value through a lookup, falling back to its raw string form. */
export function classDisplayName(value, lookup) {
  if (value === null || value === undefined) return ''
  return lookup?.get(String(value)) ?? String(value)
}
