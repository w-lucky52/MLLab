/**
 * One-vs-Rest ROC helpers.
 *
 * Pure functions only: no DOM access and no ECharts dependency, so the
 * calculations can be unit-tested or verified outside the browser.
 */

export type RocCurve = {
  classLabel: string | number
  fpr: number[]
  tpr: number[]
  auc: number
}

/** Raised when the input arrays cannot produce a trustworthy ROC curve. */
export class RocInputError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'RocInputError'
  }
}

const toFiniteScore = (value: unknown): number => {
  const score = Number(value)
  if (!Number.isFinite(score)) {
    throw new RocInputError('预测分数包含非有限值，无法绘制 ROC 曲线。')
  }
  return score
}

const sameLabel = (left: unknown, right: unknown): boolean => String(left) === String(right)

/**
 * Compute one ROC curve per class using the One-vs-Rest strategy.
 *
 * For each column of ``yScore`` the matching ``scoreClasses`` entry marks the
 * positive class; every other label is treated as negative. Classes that are
 * all-positive or all-negative in ``yTrue`` are skipped because their ROC is
 * undefined.
 *
 * Tied scores are grouped: all samples sharing a score are added to the
 * cumulative TP/FP counts before a single ROC point is emitted. This keeps
 * ``fpr`` / ``tpr`` monotonically non-decreasing and the curve ends at (1, 1).
 *
 * ``decision`` scores are used exactly like probabilities here: only their
 * ordering matters, so they never need to lie in [0, 1].
 *
 * @throws {RocInputError} when shapes disagree or a score is not finite.
 */
export function calculateOvRRoc(
  yTrue: Array<string | number>,
  yScore: number[][],
  scoreClasses: Array<string | number>,
): RocCurve[] {
  if (!Array.isArray(scoreClasses) || scoreClasses.length === 0) {
    throw new RocInputError('缺少预测分数对应的类别信息，无法绘制 ROC 曲线。')
  }
  if (!Array.isArray(yTrue) || yTrue.length === 0) {
    throw new RocInputError('缺少真实标签，无法绘制 ROC 曲线。')
  }
  if (!Array.isArray(yScore) || yScore.length !== yTrue.length) {
    throw new RocInputError('预测分数行数与真实标签数量不一致，无法绘制 ROC 曲线。')
  }
  if (yScore.some((row) => !Array.isArray(row) || row.length !== scoreClasses.length)) {
    throw new RocInputError('预测分数列数与类别数量不一致，无法绘制 ROC 曲线。')
  }

  const scores = yScore.map((row) => row.map(toFiniteScore))
  const curves: RocCurve[] = []

  scoreClasses.forEach((classLabel, column) => {
    const positive = yTrue.map((label) => (sameLabel(label, classLabel) ? 1 : 0))
    const totalPositive = positive.reduce((sum, value) => sum + value, 0)
    const totalNegative = positive.length - totalPositive

    // A class missing from either the positive or negative side has no ROC.
    if (totalPositive === 0 || totalNegative === 0) return

    const pairs = positive.map((label, index) => ({ label, score: scores[index][column] }))
    pairs.sort((left, right) => right.score - left.score)

    const fpr = [0]
    const tpr = [0]
    let truePositive = 0
    let falsePositive = 0
    let index = 0

    while (index < pairs.length) {
      const threshold = pairs[index].score
      // Group tied scores so every threshold contributes exactly one point.
      while (index < pairs.length && pairs[index].score === threshold) {
        if (pairs[index].label === 1) truePositive += 1
        else falsePositive += 1
        index += 1
      }
      fpr.push(falsePositive / totalNegative)
      tpr.push(truePositive / totalPositive)
    }

    let auc = 0
    for (let point = 1; point < fpr.length; point += 1) {
      auc += ((fpr[point] - fpr[point - 1]) * (tpr[point] + tpr[point - 1])) / 2
    }

    curves.push({ classLabel, fpr, tpr, auc })
  })

  return curves
}
