import { layeredMetricNumber, resolveAsrErrorShares } from './metricNormalization.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

const partialShares = resolveAsrErrorShares(
  { substituteShare: 0.6, insertShare: null },
  null,
  null,
)
assert(partialShares?.substituteShare === 0.6, 'available ASR error shares must be preserved')
assert(partialShares?.insertShare === null, 'a missing ASR error share must remain null instead of becoming zero')
assert(partialShares?.deleteShare === null, 'an omitted ASR error share must remain null instead of becoming zero')

const recomputedShares = resolveAsrErrorShares(
  { substituteShare: 0.5, insertShare: null },
  { substitutions: 2, insertions: 1, deletions: 1, totalErrors: 4 },
  null,
)
assert(recomputedShares?.substituteShare === 0.5, 'direct ASR error shares must take precedence')
assert(recomputedShares?.insertShare === 0.25, 'complete edit counts may fill a missing ASR error share')
assert(recomputedShares?.deleteShare === 0.25, 'complete edit counts may fill an omitted ASR error share')

const noErrors = resolveAsrErrorShares(
  null,
  { substitutions: 0, insertions: 0, deletions: 0, totalErrors: 0 },
  null,
)
assert(noErrors?.substituteShare === 0, 'zero is valid when complete edit counts prove there were no errors')

assert(
  layeredMetricNumber([{ cloneIdentityScore: null }, { cloneIdentityScore: 88 }], ['cloneIdentityScore']) === null,
  'an explicit null nested clone metric must not fall back to a stale outer value',
)
assert(
  layeredMetricNumber([{}, { cloneIdentityScore: 88 }], ['cloneIdentityScore']) === 88,
  'an absent nested clone metric may use the compatible outer value',
)
assert(
  layeredMetricNumber([{ clone_identity_score: null }, { cloneIdentityScore: 88 }], ['cloneIdentityScore', 'clone_identity_score']) === null,
  'an explicit null legacy alias must also block stale outer fallback',
)
