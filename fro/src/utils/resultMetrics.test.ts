import type { LossTrendPoint } from '../types/task.ts'
import { analyzeLossConvergence, lossConvergenceThresholds } from './resultMetrics.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

type CurveFactory = (stepIndex: number, pointCount: number) => number

function makeTrace(pointCount: number, curves: Partial<Record<'Lid' | 'Lsem' | 'Lpsy' | 'L2' | 'total', CurveFactory>>): LossTrendPoint[] {
  return Array.from({ length: pointCount }, (_, stepIndex) => ({
    step: stepIndex + 1,
    Lid: curves.Lid?.(stepIndex, pointCount) ?? null,
    Lsem: curves.Lsem?.(stepIndex, pointCount) ?? null,
    Lpsy: curves.Lpsy?.(stepIndex, pointCount) ?? null,
    L2: curves.L2?.(stepIndex, pointCount) ?? null,
    total: curves.total?.(stepIndex, pointCount) ?? null,
  }))
}

function downsampleTrace(trace: LossTrendPoint[], maxPoints = 80) {
  if (trace.length <= maxPoints) return trace
  return Array.from({ length: maxPoints }, (_, index) => trace[Math.round((index / Math.max(1, maxPoints - 1)) * (trace.length - 1))])
}

const continuing = makeTrace(80, {
  Lid: (index) => 3.1 - index * 0.024,
  Lsem: (index) => 4 - index * 0.03,
  Lpsy: (index) => index * 28,
  L2: (index) => 0.000001 + index * 0.00004,
  total: (index) => 1660 - index * 12,
})
const continuingAnalysis = analyzeLossConvergence(continuing)
assert(continuingAnalysis.status === 'unconverged', 'A majority of continuously changing late-stage curves should be marked unconverged')
assert(continuingAnalysis.active_losses.length === 5, 'All continuously changing synthetic loss curves should remain active')
assert(
  (continuingAnalysis.losses.total.final_to_reference_max_ratio ?? 0) >= lossConvergenceThresholds.minimumTailToInitialMaxSlopeRatio
    && (continuingAnalysis.losses.total.normalized_final_max_slope ?? 0) >= lossConvergenceThresholds.minimumNormalizedTailMaxSlope,
  'An active curve must satisfy both the relative and normalized robust-maximum slope thresholds',
)

function plateau(index: number, pointCount: number, start: number, change: number) {
  const plateauIndex = Math.floor(pointCount * 0.7)
  const effectiveIndex = Math.min(index, plateauIndex)
  return start + effectiveIndex * change
}

const stableTail = makeTrace(80, {
  Lid: (index, count) => plateau(index, count, 3.1, -0.03),
  Lsem: (index, count) => plateau(index, count, 4, -0.035),
  Lpsy: (index, count) => plateau(index, count, 0, 30),
  L2: (index, count) => plateau(index, count, 0.000001, 0.00004),
  total: (index, count) => plateau(index, count, 1660, -13),
})
const stableTailAnalysis = analyzeLossConvergence(stableTail)
assert(stableTailAnalysis.status === 'converged', 'Curves that flatten before the final quarter should be marked converged')
assert(stableTailAnalysis.active_losses.length === 0, 'Stable late-stage curves should not remain active')

const oneActiveCurve = makeTrace(80, {
  Lid: (index, count) => plateau(index, count, 3.1, -0.03),
  Lsem: (index, count) => plateau(index, count, 4, -0.035),
  Lpsy: (index) => index * 30,
  L2: (index, count) => plateau(index, count, 0.000001, 0.00004),
  total: (index, count) => plateau(index, count, 1660, -13),
})
const oneActiveAnalysis = analyzeLossConvergence(oneActiveCurve)
assert(oneActiveAnalysis.status === 'unconverged', 'One active curve must trigger a task-level unconverged result')
assert(
  oneActiveAnalysis.active_losses.length === 1 && oneActiveAnalysis.active_losses[0] === 'Lpsy',
  'The per-curve diagnostics should still identify the single active loss',
)

const relativeFlattening = makeTrace(80, {
  Lid: (index) => index < 20 ? 3.1 - index * 0.05 : 2.15 - (index - 20) * 0.012,
  Lsem: (index, count) => plateau(index, count, 4, -0.035),
  Lpsy: (index, count) => plateau(index, count, 0, 30),
  L2: (index, count) => plateau(index, count, 0.000001, 0.00004),
  total: (index, count) => plateau(index, count, 1660, -13),
})
const relativeFlatteningAnalysis = analyzeLossConvergence(relativeFlattening)
assert(relativeFlatteningAnalysis.status === 'converged', 'A visibly flatter tail should converge when its robust maximum slope is below the initial-relative threshold')
assert(
  (relativeFlatteningAnalysis.losses.Lid.final_to_reference_max_ratio ?? Number.POSITIVE_INFINITY)
    < lossConvergenceThresholds.minimumTailToInitialMaxSlopeRatio,
  'The same-curve initial/tail comparison should recognize relative flattening',
)

function convergingPsyCurve(index: number) {
  if (index <= 220) return index * 10
  if (index >= 320) return 2200 + 1000 / 3
  const remaining = (320 - index) / 100
  return 2200 + (1000 / 3) * (1 - remaining ** 3)
}

function makeStepCalibrationTrace(pointCount: number) {
  return makeTrace(pointCount, {
    Lid: (index, count) => plateau(index, count, 3.1, -0.01),
    Lsem: (index, count) => plateau(index, count, 4, -0.012),
    Lpsy: (index) => convergingPsyCurve(index),
    L2: (index, count) => plateau(index, count, 0.000001, 0.00001),
    total: (index, count) => plateau(index, count, 1660, -5),
  })
}

const twoHundredStepAnalysis = analyzeLossConvergence(downsampleTrace(makeStepCalibrationTrace(200)))
const threeHundredStepAnalysis = analyzeLossConvergence(downsampleTrace(makeStepCalibrationTrace(300)))
const fourHundredStepAnalysis = analyzeLossConvergence(downsampleTrace(makeStepCalibrationTrace(400)))
assert(twoHundredStepAnalysis.status === 'unconverged', 'A 200-step trace that is still climbing must recommend more iterations')
assert(threeHundredStepAnalysis.status === 'unconverged', 'A 300-step trace with a visibly active tail must still recommend more iterations')
assert(fourHundredStepAnalysis.status === 'converged', 'A 400-step trace whose final quarter has flattened should be marked converged')
assert(
  (threeHundredStepAnalysis.losses.Lpsy.final_to_reference_max_ratio ?? 0) >= lossConvergenceThresholds.minimumTailToInitialMaxSlopeRatio
    && (fourHundredStepAnalysis.losses.Lpsy.final_to_reference_max_ratio ?? Number.POSITIVE_INFINITY) < lossConvergenceThresholds.minimumTailToInitialMaxSlopeRatio,
  'The calibrated relative-slope threshold must separate the 300-step active tail from the 400-step plateau',
)

const twoHundredStepPlateau = makeTrace(200, {
  Lid: (index, count) => plateau(index, count, 3.1, -0.01),
  Lsem: (index, count) => plateau(index, count, 4, -0.012),
  Lpsy: (index, count) => plateau(index, count, 0, 12),
  L2: (index, count) => plateau(index, count, 0.000001, 0.00001),
  total: (index, count) => plateau(index, count, 1660, -5),
})
const twoHundredStepPlateauAnalysis = analyzeLossConvergence(downsampleTrace(twoHundredStepPlateau))
assert(twoHundredStepPlateauAnalysis.status === 'converged', 'A 200-step trace with all five curves plateaued before the final quarter should remain converged')

const insufficient = analyzeLossConvergence(continuing.slice(0, 31))
assert(insufficient.status === 'insufficient', 'Fewer than 32 points must not produce a convergence conclusion')
assert(insufficient.valid_loss_count === 0, 'All short curves should remain explicitly insufficient')

console.log('resultMetrics tests passed')
