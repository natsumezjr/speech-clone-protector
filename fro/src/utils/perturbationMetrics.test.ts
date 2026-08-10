import { PCM16_QUANTIZATION_STEP, resolveEpsilonUsageRate } from './perturbationMetrics.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

const epsilon = 4 / 255
const quantizedRate = (epsilon + PCM16_QUANTIZATION_STEP * 0.99) / epsilon
assert(
  resolveEpsilonUsageRate({ epsilon, epsilonNorm: 'linf', epsilonUsageRate: quantizedRate }) === 1,
  'one PCM16 quantization step must display as 100% utilization',
)

const realOverrunRate = (epsilon + PCM16_QUANTIZATION_STEP * 1.5) / epsilon
assert(
  resolveEpsilonUsageRate({ epsilon, epsilonNorm: 'linf', epsilonUsageRate: realOverrunRate }) === realOverrunRate,
  'a real overrun beyond serialization tolerance must remain visible',
)

assert(
  resolveEpsilonUsageRate({ epsilon: 2, epsilonNorm: 'l2', l2Norm: 3 }) === 1.5,
  'L2 utilization must not use the PCM16 L-infinity tolerance',
)
