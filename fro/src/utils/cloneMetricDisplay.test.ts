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
assert(
  visibleText.includes('声纹相似度由 0.45 降至 -0.04，下降 0.49；声纹嵌入距离由 0.55 升至 1.04，增大 0.49，说明保护后的克隆声音与原说话人更不相似，两个声纹嵌入向量的夹角更大。'),
  'Similarity decrease and cosine-distance increase should be merged into one concrete identity conclusion',
)
assert(!visibleText.includes('保护后声纹相似度的值为'), 'Protected similarity should no longer be emitted as a separate insight')
assert(!visibleText.includes('保护后声纹距离的值为'), 'Protected distance should no longer be emitted as a separate insight')

const cloneScoreItems = generateCloneMetricInsights({
  originalSimilarity: 0.60,
  protectedSimilarity: 0.35,
  embeddingDistanceBefore: 0.40,
  embeddingDistanceAfter: 0.65,
  cloneIdentityScore: 85,
  cloneSemanticScore: 84.99,
  cleanCloneTextError: 0.2026,
  protectedCloneTextError: 0.3108,
  cloneTextChangeRate: 0.1728,
  cloneQualityScore: 90.32,
  cloneQualityRawScore: 0,
  cloneQualityRelevance: 0.0968,
  cloneQualityDropRate: 0,
})
const cloneScores = cloneScoreItems.join(' ')
assert(cloneScores.includes('声纹相似度由 0.60 降至 0.35，下降 0.25；声纹嵌入距离由 0.40 升至 0.65，增大 0.25'), 'Complete identity values should remain in one merged insight')
assert(cloneScores.includes('克隆身份保护评分的值为 85.00 分，说明克隆身份保护效果优秀。'), 'Clone identity score from 85 should be rated excellent')
assert(cloneScores.includes('克隆后语义干扰评分的值为 84.99 分，说明克隆语义保护效果中等。'), 'Clone semantic score below 85 should be rated medium')
assert(!cloneScores.includes('原始克隆文本误差'), 'Original clone text error must stay out of the interpretation list')
assert(!cloneScores.includes('保护后克隆文本误差'), 'Protected clone text error must stay out of the interpretation list')
assert(!cloneScores.includes('克隆文本变化率'), 'Clone text change rate must stay out of the interpretation list')
assert(cloneScores.includes('克隆音频质量退化评分的值为 90.32 分，说明听感质量保护效果优秀，保护之后在听感、身份、语义综合保护上达到了中等效果。'), 'Quality score and medium comprehensive conclusion should share one sentence')
assert(cloneScoreItems.filter((item) => item.includes('克隆音频质量退化评分')).length === 1, 'Quality and comprehensive interpretation should occupy one item')
assert(!cloneScoreItems.some((item) => item.startsWith('保护之后在听感、身份、语义')), 'Comprehensive conclusion must not be emitted as a separate item')
assert(!cloneScores.includes('原始质量退化分'), 'Clone insight should not explain the raw quality-score calculation')
assert(!cloneScores.includes('参考占比'), 'Clone insight should not explain the dynamic relevance calculation')

const boundaryLevels = generateCloneMetricInsights({
  originalSimilarity: 0.44,
  protectedSimilarity: 0.46,
  embeddingDistanceBefore: 0.56,
  embeddingDistanceAfter: 0.54,
  cloneIdentityScore: 69.99,
  cloneSemanticScore: 70,
  cloneQualityScore: 84.99,
  cloneTextChangeRate: 0.09,
}).join(' ')
assert(
  boundaryLevels.includes('声纹相似度由 0.44 升至 0.46，上升 0.02；声纹嵌入距离由 0.56 降至 0.54，减小 0.02，说明保护后的克隆声音与原说话人更相似，两个声纹嵌入向量的夹角更小。'),
  'Similarity increase and cosine-distance decrease should explain greater identity similarity and a smaller angle',
)
assert(boundaryLevels.includes('克隆身份保护评分的值为 69.99 分，说明克隆身份保护效果较差。'), 'Clone identity score below 70 should be rated poor')
assert(boundaryLevels.includes('克隆后语义干扰评分的值为 70.00 分，说明克隆语义保护效果中等。'), 'Clone semantic score from 70 should be rated medium')
assert(boundaryLevels.includes('克隆音频质量退化评分的值为 84.99 分，说明听感质量保护效果中等，保护之后在听感、身份、语义综合保护上的效果较差。'), 'Quality score and poor comprehensive conclusion should share one sentence')
assert(!boundaryLevels.includes('克隆文本变化率'), 'Clone text metrics must not reappear in a poor-score interpretation')

const unchanged = generateCloneMetricInsights({
  originalSimilarity: 0.35,
  protectedSimilarity: 0.35,
  embeddingDistanceBefore: 0.65,
  embeddingDistanceAfter: 0.65,
  cloneIdentityScore: 85,
  cloneSemanticScore: 85,
  cloneQualityScore: 92.45,
}).join(' ')
assert(
  unchanged.includes('声纹相似度由 0.35 变为 0.35，无明显变化（变化量 0.00）；声纹嵌入距离由 0.65 变为 0.65，无明显变化（变化量 0.00），说明保护后的克隆声音与原说话人的相似程度基本不变，两个声纹嵌入向量的夹角基本不变。'),
  'Unchanged identity metrics should not claim a direction',
)
assert(
  unchanged.includes('克隆音频质量退化评分的值为 92.45 分，说明听感质量保护效果优秀，保护之后在听感、身份、语义综合保护上达到了良好的效果。'),
  'Three excellent scores should use the exact combined quality and comprehensive sentence',
)

const incompleteIdentity = generateCloneMetricInsights({
  originalSimilarity: 0.50,
  embeddingDistanceBefore: 0.50,
  embeddingDistanceAfter: 0.75,
  cloneIdentityScore: 90,
  cloneSemanticScore: 90,
}).join(' ')
assert(
  incompleteIdentity.includes('声纹相似度前后值尚未完整生成；声纹嵌入距离由 0.50 升至 0.75，增大 0.25，说明保护后的克隆声音与原说话人更不相似，两个声纹嵌入向量的夹角更大。'),
  'A missing similarity pair should be reported while a complete cosine-distance pair remains interpretable',
)
assert(incompleteIdentity.includes('克隆音频质量退化评分尚未生成，身份、语义或听感评分尚未完整生成，暂不能给出综合保护结论。'), 'Missing quality score must use one honest combined status sentence')

const incompleteCompositeScores = generateCloneMetricInsights({
  cloneIdentityScore: 90,
  cloneQualityScore: 92.45,
}).filter((item) => item.includes('克隆音频质量退化评分'))
assert(incompleteCompositeScores.length === 1, 'A present quality score and missing semantic score should still use one combined item')
assert(
  incompleteCompositeScores[0] === '克隆音频质量退化评分的值为 92.45 分，说明听感质量保护效果优秀，但身份或语义评分尚未完整生成，暂不能给出综合保护结论。',
  'A missing identity or semantic score must prevent an excellent comprehensive conclusion',
)

const inconsistentIdentity = generateCloneMetricInsights({
  originalSimilarity: 0.50,
  protectedSimilarity: 0.30,
  embeddingDistanceBefore: 0.50,
  embeddingDistanceAfter: 0.30,
}).join(' ')
assert(
  inconsistentIdentity.includes('两项变化方向不一致，暂不能判断保护后的克隆声音与原说话人的相似程度及声纹嵌入向量夹角变化。'),
  'Contradictory similarity and distance directions must not be forced into an identity conclusion',
)

const unavailable = generateCloneMetricInsights({
  cloneSemanticReason: 'missing transcription',
  cloneQualityReason: 'missing DNSMOS',
})
assert(unavailable.includes('克隆后语义干扰评分尚未生成。'), 'Missing clone semantic score must stay visibly unavailable')
assert(unavailable.includes('克隆音频质量退化评分尚未生成，身份、语义或听感评分尚未完整生成，暂不能给出综合保护结论。'), 'Missing quality and comprehensive scores must share one honest sentence')
assert(unavailable.includes('声纹相似度前后值尚未完整生成；声纹距离前后值尚未完整生成。'), 'Missing identity pairs must stay visibly unavailable')
assert(unavailable.filter((item) => item.includes('克隆音频质量退化评分')).length === 1, 'Missing quality and comprehensive status should occupy one item')
