import type { LossTrendPoint } from '@/types/task'

export type TrendMetricKey = 'Lid' | 'Lsem' | 'Lpsy' | 'L2' | 'total'
export type TrendDirection = 'up' | 'down' | 'stable' | 'insufficient'

export interface TrendAnalysis {
  direction: TrendDirection
  relative_change: number | null
  normalized_slope: number | null
  spearman: number | null
  start_value: number | null
  end_value: number | null
  point_count: number
  window_size: number | null
  raw_values: number[]
  smoothed_values: number[]
}

export type ConvergenceStatus = 'converged' | 'unconverged' | 'insufficient'

export interface LossSlopeAnalysis {
  key: TrendMetricKey
  status: ConvergenceStatus
  reference_mean_abs_slope: number | null
  final_mean_abs_slope: number | null
  final_to_reference_ratio: number | null
  reference_max_abs_slope: number | null
  final_max_abs_slope: number | null
  final_to_reference_max_ratio: number | null
  normalized_final_slope: number | null
  normalized_final_max_slope: number | null
  endpoint_change_rate: number | null
  robust_scale: number | null
  tail_point_count: number
  segment_count: number
}

export interface ConvergenceAnalysis {
  status: ConvergenceStatus
  active_losses: TrendMetricKey[]
  valid_loss_count: number
  losses: Record<TrendMetricKey, LossSlopeAnalysis>
}

const convergenceKeys: TrendMetricKey[] = ['Lid', 'Lsem', 'Lpsy', 'L2', 'total']
// Calibrated against real 200/300/400-step traces after the same 80-point
// display downsampling used by ResultsPage. The still-changing examples had
// ratio >= 0.74 and normalized tail maximum >= 0.22, while the plateaued
// examples stayed at ratio <= 0.34 and normalized tail maximum <= 0.15.
export const lossConvergenceThresholds = {
  minimumPoints: 32,
  segmentFraction: 0.25,
  minimumSegmentPoints: 8,
  endpointSegmentFraction: 0.2,
  robustMaximumQuantile: 0.9,
  minimumTailToInitialMaxSlopeRatio: 0.35,
  minimumNormalizedTailMaxSlope: 0.18,
  minimumValidLossCountForConverged: 3,
} as const

function finiteNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value !== 'string' || value.trim() === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function metricValue(point: LossTrendPoint, key: TrendMetricKey) {
  if (key === 'Lid') return finiteNumber(point.Lid) ?? finiteNumber(point.Lfeat)
  return finiteNumber(point[key])
}

function cleanLossPoints(points: LossTrendPoint[] | null | undefined, key: TrendMetricKey) {
  return (points ?? [])
    .map((point, index) => ({
      step: finiteNumber(point.step) ?? index,
      value: metricValue(point, key),
      index,
    }))
    .filter((item): item is { step: number; value: number; index: number } => item.value !== null)
    .sort((left, right) => left.step - right.step || left.index - right.index)
}

function mean(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
}

function median(values: number[]) {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

function quantile(values: number[], q: number) {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const position = (sorted.length - 1) * q
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  if (lower === upper) return sorted[lower]
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
}

function rollingMedian(values: number[], windowSize: number) {
  const radius = Math.floor(windowSize / 2)
  return values.map((_, index) => median(values.slice(Math.max(0, index - radius), Math.min(values.length, index + radius + 1))))
}

function averageRanks(values: number[]) {
  const indexed = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value)
  const ranks = Array<number>(values.length)
  let cursor = 0
  while (cursor < indexed.length) {
    let end = cursor + 1
    while (end < indexed.length && indexed[end].value === indexed[cursor].value) end += 1
    const averageRank = (cursor + 1 + end) / 2
    for (let index = cursor; index < end; index += 1) ranks[indexed[index].index] = averageRank
    cursor = end
  }
  return ranks
}

function pearson(left: number[], right: number[]) {
  if (left.length !== right.length || left.length < 2) return 0
  const leftMean = left.reduce((sum, value) => sum + value, 0) / left.length
  const rightMean = right.reduce((sum, value) => sum + value, 0) / right.length
  let numerator = 0
  let leftScale = 0
  let rightScale = 0
  for (let index = 0; index < left.length; index += 1) {
    const leftDelta = left[index] - leftMean
    const rightDelta = right[index] - rightMean
    numerator += leftDelta * rightDelta
    leftScale += leftDelta * leftDelta
    rightScale += rightDelta * rightDelta
  }
  const denominator = Math.sqrt(leftScale * rightScale)
  return denominator > 0 ? numerator / denominator : 0
}

function theilSenSlope(x: number[], y: number[]) {
  const slopes: number[] = []
  for (let left = 0; left < x.length - 1; left += 1) {
    for (let right = left + 1; right < x.length; right += 1) {
      const dx = x[right] - x[left]
      if (Math.abs(dx) > 1e-12) slopes.push((y[right] - y[left]) / dx)
    }
  }
  return slopes.length ? median(slopes) : 0
}

export function analyzeLossTrend(points: LossTrendPoint[] | null | undefined, key: TrendMetricKey): TrendAnalysis {
  const cleaned = cleanLossPoints(points, key)

  const rawValues = cleaned.map((item) => item.value)
  if (cleaned.length < 6) {
    return {
      direction: 'insufficient',
      relative_change: null,
      normalized_slope: null,
      spearman: null,
      start_value: null,
      end_value: null,
      point_count: cleaned.length,
      window_size: null,
      raw_values: rawValues,
      smoothed_values: [],
    }
  }

  let windowSize = Math.max(3, Math.round(cleaned.length * 0.05))
  if (windowSize % 2 === 0) windowSize += 1
  const smoothed = rollingMedian(rawValues, windowSize)
  const segmentSize = Math.max(1, Math.ceil(smoothed.length * 0.2))
  const startValue = median(smoothed.slice(0, segmentSize))
  const endValue = median(smoothed.slice(-segmentSize))
  const iqr = quantile(smoothed, 0.75) - quantile(smoothed, 0.25)
  const scale = Math.max(Math.abs(startValue), Math.abs(iqr), 1e-8)
  const relativeChange = (endValue - startValue) / scale

  const steps = cleaned.map((item) => item.step)
  const minStep = Math.min(...steps)
  const maxStep = Math.max(...steps)
  const normalizedSteps = maxStep > minStep
    ? steps.map((step) => (step - minStep) / (maxStep - minStep))
    : steps.map((_, index) => index / Math.max(1, steps.length - 1))
  const normalizedSlope = theilSenSlope(normalizedSteps, smoothed) / scale
  const spearman = pearson(averageRanks(normalizedSteps), averageRanks(smoothed))

  let direction: TrendDirection = 'stable'
  if (relativeChange >= 0.05 && normalizedSlope >= 0.05 && spearman >= 0.35) direction = 'up'
  if (relativeChange <= -0.05 && normalizedSlope <= -0.05 && spearman <= -0.35) direction = 'down'

  return {
    direction,
    relative_change: relativeChange,
    normalized_slope: normalizedSlope,
    spearman,
    start_value: startValue,
    end_value: endValue,
    point_count: cleaned.length,
    window_size: windowSize,
    raw_values: rawValues,
    smoothed_values: smoothed,
  }
}

function analyzeLossSlope(points: LossTrendPoint[] | null | undefined, key: TrendMetricKey): LossSlopeAnalysis {
  const cleaned = cleanLossPoints(points, key)
  const insufficient = (segmentCount = 0): LossSlopeAnalysis => ({
    key,
    status: 'insufficient',
    reference_mean_abs_slope: null,
    final_mean_abs_slope: null,
    final_to_reference_ratio: null,
    reference_max_abs_slope: null,
    final_max_abs_slope: null,
    final_to_reference_max_ratio: null,
    normalized_final_slope: null,
    normalized_final_max_slope: null,
    endpoint_change_rate: null,
    robust_scale: null,
    tail_point_count: 0,
    segment_count: segmentCount,
  })
  if (cleaned.length < lossConvergenceThresholds.minimumPoints) {
    return insufficient(Math.max(0, cleaned.length - 1))
  }

  let smoothingWindow = Math.max(3, Math.round(cleaned.length * 0.05))
  if (smoothingWindow % 2 === 0) smoothingWindow += 1
  const steps = cleaned.map((item) => item.step)
  const smoothedValues = rollingMedian(cleaned.map((item) => item.value), smoothingWindow)
  const tailPointCount = Math.max(
    lossConvergenceThresholds.minimumSegmentPoints,
    Math.ceil(smoothedValues.length * lossConvergenceThresholds.segmentFraction),
  )
  const initialSteps = steps.slice(0, tailPointCount)
  const initialValues = smoothedValues.slice(0, tailPointCount)
  const tailSteps = steps.slice(-tailPointCount)
  const tailValues = smoothedValues.slice(-tailPointCount)
  const initialStepSpan = initialSteps.at(-1)! - initialSteps[0]
  const tailStepSpan = tailSteps.at(-1)! - tailSteps[0]
  if (
    initialValues.length < lossConvergenceThresholds.minimumSegmentPoints
    || tailValues.length < lossConvergenceThresholds.minimumSegmentPoints
    || initialStepSpan <= 1e-12
    || tailStepSpan <= 1e-12
  ) {
    return insufficient(Math.max(0, cleaned.length - 1))
  }

  const robustScale = Math.max(
    quantile(smoothedValues, 0.95) - quantile(smoothedValues, 0.05),
    Math.abs(median(smoothedValues)) * 0.05,
    1e-8,
  )
  const robustTailSlope = theilSenSlope(tailSteps, tailValues)
  const normalizedFinalSlope = Math.abs(robustTailSlope) * tailStepSpan / robustScale
  const endpointPointCount = Math.max(2, Math.ceil(tailValues.length * lossConvergenceThresholds.endpointSegmentFraction))
  const endpointChangeRate = Math.abs(
    median(tailValues.slice(-endpointPointCount)) - median(tailValues.slice(0, endpointPointCount)),
  ) / robustScale

  const localSlopes = (segmentSteps: number[], segmentValues: number[]) => {
    const slopes: number[] = []
    for (let index = 1; index < segmentValues.length; index += 1) {
      const stepDelta = segmentSteps[index] - segmentSteps[index - 1]
      if (Math.abs(stepDelta) <= 1e-12) continue
      slopes.push(Math.abs((segmentValues[index] - segmentValues[index - 1]) / stepDelta))
    }
    return slopes
  }
  const slopes: number[] = []
  for (let index = 1; index < smoothedValues.length; index += 1) {
    const stepDelta = steps[index] - steps[index - 1]
    if (Math.abs(stepDelta) <= 1e-12) continue
    slopes.push(Math.abs((smoothedValues[index] - smoothedValues[index - 1]) / stepDelta))
  }
  if (slopes.length < lossConvergenceThresholds.minimumPoints - 1) {
    return insufficient(slopes.length)
  }

  const referenceSlopes = localSlopes(initialSteps, initialValues)
  const finalSlopes = localSlopes(tailSteps, tailValues)
  if (referenceSlopes.length === 0 || finalSlopes.length === 0) {
    return insufficient(slopes.length)
  }

  const referenceSlope = mean(referenceSlopes)
  const finalSlope = mean(finalSlopes)
  const slopeRatio = referenceSlope > 1e-12
    ? finalSlope / referenceSlope
    : finalSlope <= 1e-12
      ? 0
      : Number.POSITIVE_INFINITY
  // A literal maximum is overly sensitive to a single noisy step. The 90th
  // percentile is used as a robust maximum while still reflecting the steepest
  // visible part of the initial and final segments of the same curve.
  const referenceMaxSlope = quantile(referenceSlopes, lossConvergenceThresholds.robustMaximumQuantile)
  const finalMaxSlope = quantile(finalSlopes, lossConvergenceThresholds.robustMaximumQuantile)
  const maxSlopeRatio = referenceMaxSlope > 1e-12
    ? finalMaxSlope / referenceMaxSlope
    : finalMaxSlope <= 1e-12
      ? 0
      : Number.POSITIVE_INFINITY
  const normalizedFinalMaxSlope = finalMaxSlope * tailStepSpan / robustScale
  const remainsActive = maxSlopeRatio >= lossConvergenceThresholds.minimumTailToInitialMaxSlopeRatio
    && normalizedFinalMaxSlope >= lossConvergenceThresholds.minimumNormalizedTailMaxSlope

  return {
    key,
    status: remainsActive ? 'unconverged' : 'converged',
    reference_mean_abs_slope: referenceSlope,
    final_mean_abs_slope: finalSlope,
    final_to_reference_ratio: slopeRatio,
    reference_max_abs_slope: referenceMaxSlope,
    final_max_abs_slope: finalMaxSlope,
    final_to_reference_max_ratio: maxSlopeRatio,
    normalized_final_slope: normalizedFinalSlope,
    normalized_final_max_slope: normalizedFinalMaxSlope,
    endpoint_change_rate: endpointChangeRate,
    robust_scale: robustScale,
    tail_point_count: tailPointCount,
    segment_count: slopes.length,
  }
}

export function analyzeLossConvergence(points: LossTrendPoint[] | null | undefined): ConvergenceAnalysis {
  const losses = Object.fromEntries(
    convergenceKeys.map((key) => [key, analyzeLossSlope(points, key)]),
  ) as Record<TrendMetricKey, LossSlopeAnalysis>
  const validLosses = convergenceKeys.filter((key) => losses[key].status !== 'insufficient')
  const activeLosses = validLosses.filter((key) => losses[key].status === 'unconverged')
  if (activeLosses.length > 0) {
    return { status: 'unconverged', active_losses: activeLosses, valid_loss_count: validLosses.length, losses }
  }
  if (validLosses.length < lossConvergenceThresholds.minimumValidLossCountForConverged) {
    return { status: 'insufficient', active_losses: activeLosses, valid_loss_count: validLosses.length, losses }
  }
  return {
    status: 'converged',
    active_losses: activeLosses,
    valid_loss_count: validLosses.length,
    losses,
  }
}
