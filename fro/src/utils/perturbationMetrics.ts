export type EpsilonUsageInput = {
  linfNorm?: number | null
  l2Norm?: number | null
  epsilon?: number | null
  epsilonNorm?: string | null
  epsilonUsageRate?: number | null
  epsilonUsageRateRaw?: number | null
  epsilonToleranceRate?: number | null
  epsilonExceeded?: boolean | null
}

export const PCM16_QUANTIZATION_STEP = 1 / 32768

function finiteNumber(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function rawEpsilonUsageRate(input?: EpsilonUsageInput | null) {
  if (!input) return null
  const explicitRaw = finiteNumber(input.epsilonUsageRateRaw)
  if (explicitRaw !== null) return explicitRaw
  const explicit = finiteNumber(input.epsilonUsageRate)
  if (explicit !== null) return explicit
  const epsilon = finiteNumber(input.epsilon)
  if (epsilon === null || epsilon <= 0) return null
  if (String(input.epsilonNorm ?? '').toLowerCase() === 'linf') {
    const linfNorm = finiteNumber(input.linfNorm)
    return linfNorm === null ? null : linfNorm / epsilon
  }
  if (String(input.epsilonNorm ?? '').toLowerCase() === 'l2') {
    const l2Norm = finiteNumber(input.l2Norm)
    return l2Norm === null ? null : l2Norm / epsilon
  }
  return null
}

/**
 * Returns the user-facing budget utilization ratio.
 *
 * A PCM16 round trip may enlarge one sample by one quantization step after
 * the optimizer has already projected the float-domain perturbation to the
 * configured L-infinity limit. That narrow serialization tolerance is shown
 * as exactly 100%; larger overruns remain visible instead of being hidden.
 */
export function resolveEpsilonUsageRate(input?: EpsilonUsageInput | null) {
  const raw = rawEpsilonUsageRate(input)
  if (raw === null) return null
  if (input?.epsilonExceeded === true) return raw
  if (input?.epsilonExceeded === false) return Math.min(raw, 1)
  if (String(input?.epsilonNorm ?? '').toLowerCase() !== 'linf') return raw
  const epsilon = finiteNumber(input?.epsilon)
  if (epsilon === null || epsilon <= 0) return raw
  const tolerance = finiteNumber(input?.epsilonToleranceRate) ?? PCM16_QUANTIZATION_STEP / epsilon
  return raw <= 1 + tolerance + Number.EPSILON ? Math.min(raw, 1) : raw
}
