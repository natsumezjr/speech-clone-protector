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
  const cleaned = (points ?? [])
    .map((point, index) => ({
      step: finiteNumber(point.step) ?? index,
      value: metricValue(point, key),
      index,
    }))
    .filter((item): item is { step: number; value: number; index: number } => item.value !== null)
    .sort((left, right) => left.step - right.step || left.index - right.index)

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
