import {
  cloneMetricDisplay,
  computeAbsoluteDelta,
  computeAbsoluteDrop,
  generateCloneMetricInsights,
} from './cloneMetricDisplay.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

function assertClose(actual: number | null, expected: number, message: string) {
  assert(actual !== null && Math.abs(actual - expected) < 1.0e-9, message)
}

const sample = {
  originalSimilarity: 0.449,
  protectedSimilarity: -0.042,
  embeddingDistanceBefore: 0.551,
  embeddingDistanceAfter: 1.042,
}

const display = cloneMetricDisplay(sample)
assert(display.similarityBefore === '0.45', 'Speaker Similarity before should keep two decimals')
assert(display.similarityAfter === '-0.04', 'Negative protected similarity must not be clamped')
assert(display.similarityDeltaText === '下降 0.49', 'Speaker Similarity delta should be an absolute decrease')
assert(display.embeddingDistanceBefore === '0.55', 'Embedding distance before should keep two decimals')
assert(display.embeddingDistanceAfter === '1.04', 'Embedding distance above 1 must not be clamped')
assert(display.embeddingDistanceDeltaText === '增加 0.49', 'Embedding distance delta should be an absolute increase')
assert(display.similarityDropAbsText === '0.49', 'Similarity drop amount should be absolute, not percent')
assertClose(computeAbsoluteDelta(sample.originalSimilarity, sample.protectedSimilarity), -0.491, 'Absolute delta should be after - before')
assertClose(computeAbsoluteDrop(sample.originalSimilarity, sample.protectedSimilarity), 0.491, 'Absolute drop should be before - after')

const explicitDistanceDelta = cloneMetricDisplay({
  ...sample,
  embeddingDistanceDelta: 0.25,
})
assert(explicitDistanceDelta.embeddingDistanceDeltaText === '增加 0.25', 'Explicit backend embedding distance delta should take precedence')

const visibleText = [
  display.similarityBefore,
  display.similarityAfter,
  display.similarityDeltaText,
  display.embeddingDistanceBefore,
  display.embeddingDistanceAfter,
  display.embeddingDistanceDeltaText,
  display.similarityDropAbsText,
  ...generateCloneMetricInsights(sample),
].join(' ')

assert(!visibleText.includes('↓ 100.0%'), 'Speaker Similarity card must not show relative percent drop')
assert(!visibleText.includes('↑ 89.1%'), 'Embedding distance card must not show relative percent increase')
assert(!visibleText.includes('防护下降率'), 'Clone result label should be replaced by 相似度下降量 in UI')
