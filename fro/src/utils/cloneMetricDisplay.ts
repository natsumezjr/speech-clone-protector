export type CloneMetricInput = {
  originalSimilarity?: number | null
  protectedSimilarity?: number | null
  embeddingDistanceBefore?: number | null
  embeddingDistanceAfter?: number | null
  embeddingDistanceDelta?: number | null
  cloneIdentityScore?: number | null
  cloneSemanticScore?: number | null
  cloneQualityRawScore?: number | null
  cloneQualityRelevance?: number | null
  cloneQualityScore?: number | null
  cloneQualityDropRate?: number | null
  cleanCloneTextError?: number | null
  protectedCloneTextError?: number | null
  cloneTextChangeRate?: number | null
  cloneSemanticStatus?: string | null
  cloneSemanticReason?: string | null
  cloneQualityStatus?: string | null
  cloneQualityReason?: string | null
}

export function optionalMetricNumber(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

export function computeAbsoluteDelta(before: number | null | undefined, after: number | null | undefined) {
  const beforeValue = optionalMetricNumber(before)
  const afterValue = optionalMetricNumber(after)
  if (beforeValue === null || afterValue === null) return null
  return afterValue - beforeValue
}

export function computeAbsoluteDrop(before: number | null | undefined, after: number | null | undefined) {
  const beforeValue = optionalMetricNumber(before)
  const afterValue = optionalMetricNumber(after)
  if (beforeValue === null || afterValue === null) return null
  return beforeValue - afterValue
}

export function formatCloneMetricNumber(value: unknown) {
  const numberValue = optionalMetricNumber(value)
  if (numberValue === null) return '未生成'
  return numberValue.toFixed(2)
}

export function formatSimilarityDelta(delta: number | null | undefined) {
  const value = optionalMetricNumber(delta)
  if (value === null) return '未生成'
  if (value < 0) return `下降 ${Math.abs(value).toFixed(2)}`
  if (value > 0) return `上升 ${value.toFixed(2)}`
  return '无明显变化'
}

export function formatEmbeddingDistanceDelta(delta: number | null | undefined) {
  const value = optionalMetricNumber(delta)
  if (value === null) return '未生成'
  if (value > 0) return `增加 ${value.toFixed(2)}`
  if (value < 0) return `减少 ${Math.abs(value).toFixed(2)}`
  return '无明显变化'
}

export function cloneMetricDisplay(input: CloneMetricInput) {
  const similarityDelta = computeAbsoluteDelta(input.originalSimilarity, input.protectedSimilarity)
  const similarityDropAbs = computeAbsoluteDrop(input.originalSimilarity, input.protectedSimilarity)
  const embeddingDistanceDelta = optionalMetricNumber(input.embeddingDistanceDelta)
    ?? computeAbsoluteDelta(input.embeddingDistanceBefore, input.embeddingDistanceAfter)
  return {
    similarityBefore: formatCloneMetricNumber(input.originalSimilarity),
    similarityAfter: formatCloneMetricNumber(input.protectedSimilarity),
    similarityDeltaText: formatSimilarityDelta(similarityDelta),
    similarityDropAbs,
    similarityDropAbsText: formatCloneMetricNumber(similarityDropAbs),
    embeddingDistanceBefore: formatCloneMetricNumber(input.embeddingDistanceBefore),
    embeddingDistanceAfter: formatCloneMetricNumber(input.embeddingDistanceAfter),
    embeddingDistanceDeltaText: formatEmbeddingDistanceDelta(embeddingDistanceDelta),
  }
}

function higherIsBetterLevel(value: number, mediumThreshold: number, excellentThreshold: number) {
  if (value >= excellentThreshold) return '优秀'
  if (value >= mediumThreshold) return '中等'
  return '较差'
}

type ChangeDirection = 'up' | 'down' | 'stable'

function changeDirection(delta: number): ChangeDirection {
  if (Math.abs(delta) < 0.005) return 'stable'
  return delta > 0 ? 'up' : 'down'
}

function changeText(label: string, before: number, after: number, kind: 'similarity' | 'distance') {
  const delta = after - before
  const direction = changeDirection(delta)
  if (direction === 'stable') return `${label}由 ${before.toFixed(2)} 变为 ${after.toFixed(2)}，无明显变化（变化量 0.00）`
  const transition = direction === 'up' ? '升至' : '降至'
  const change = kind === 'distance'
    ? direction === 'up' ? '增大' : '减小'
    : direction === 'up' ? '上升' : '下降'
  return `${label}由 ${before.toFixed(2)} ${transition} ${after.toFixed(2)}，${change} ${Math.abs(delta).toFixed(2)}`
}

function identityDirectionConclusion(similarityDirection: ChangeDirection | null, distanceDirection: ChangeDirection | null) {
  if (similarityDirection !== null && distanceDirection !== null) {
    if (similarityDirection === 'down' && distanceDirection === 'up') return '说明保护后的克隆声音与原说话人更不相似，两个声纹嵌入向量的夹角更大。'
    if (similarityDirection === 'up' && distanceDirection === 'down') return '说明保护后的克隆声音与原说话人更相似，两个声纹嵌入向量的夹角更小。'
    if (similarityDirection === 'stable' && distanceDirection === 'stable') return '说明保护后的克隆声音与原说话人的相似程度基本不变，两个声纹嵌入向量的夹角基本不变。'
    return '两项变化方向不一致，暂不能判断保护后的克隆声音与原说话人的相似程度及声纹嵌入向量夹角变化。'
  }
  if (similarityDirection === 'down') return '说明保护后的克隆声音与原说话人更不相似。'
  if (similarityDirection === 'up') return '说明保护后的克隆声音与原说话人更相似。'
  if (similarityDirection === 'stable') return '说明保护后的克隆声音与原说话人的相似程度基本不变。'
  if (distanceDirection === 'up') return '说明保护后的克隆声音与原说话人更不相似，两个声纹嵌入向量的夹角更大。'
  if (distanceDirection === 'down') return '说明保护后的克隆声音与原说话人更相似，两个声纹嵌入向量的夹角更小。'
  if (distanceDirection === 'stable') return '说明保护后的克隆声音与原说话人的相似程度基本不变，两个声纹嵌入向量的夹角基本不变。'
  return ''
}

function identityChangeInsight(input: CloneMetricInput) {
  const originalSimilarity = optionalMetricNumber(input.originalSimilarity)
  const protectedSimilarity = optionalMetricNumber(input.protectedSimilarity)
  const distanceBefore = optionalMetricNumber(input.embeddingDistanceBefore)
  const distanceAfter = optionalMetricNumber(input.embeddingDistanceAfter)
  const similarityComplete = originalSimilarity !== null && protectedSimilarity !== null
  const distanceComplete = distanceBefore !== null && distanceAfter !== null
  const details: string[] = []
  let similarityDirection: ChangeDirection | null = null
  let distanceDirection: ChangeDirection | null = null

  if (similarityComplete) {
    similarityDirection = changeDirection(protectedSimilarity - originalSimilarity)
    details.push(changeText('声纹相似度', originalSimilarity, protectedSimilarity, 'similarity'))
  } else {
    details.push('声纹相似度前后值尚未完整生成')
  }
  if (distanceComplete) {
    distanceDirection = changeDirection(distanceAfter - distanceBefore)
    details.push(changeText('声纹嵌入距离', distanceBefore, distanceAfter, 'distance'))
  } else {
    details.push('声纹距离前后值尚未完整生成')
  }

  // 后端 compute_clone_identity_score 明确定义 embeddingDistance = 1 - cosineSimilarity，
  // 因此距离增减可用于解释两个声纹向量夹角的增减。
  const conclusion = identityDirectionConclusion(similarityDirection, distanceDirection)
  return `${details.join('；')}${conclusion ? `，${conclusion}` : '。'}`
}

export function generateCloneMetricInsights(input: CloneMetricInput) {
  const identityScore = optionalMetricNumber(input.cloneIdentityScore)
  const semanticScore = optionalMetricNumber(input.cloneSemanticScore)
  const qualityScore = optionalMetricNumber(input.cloneQualityScore)
  const items: string[] = [identityChangeInsight(input)]

  if (identityScore !== null) {
    items.push(`克隆身份保护评分的值为 ${identityScore.toFixed(2)} 分，说明克隆身份保护效果${higherIsBetterLevel(identityScore, 70, 85)}。`)
  }
  if (semanticScore !== null) {
    items.push(`克隆后语义干扰评分的值为 ${semanticScore.toFixed(2)} 分，说明克隆语义保护效果${higherIsBetterLevel(semanticScore, 70, 85)}。`)
  } else if (input.cloneSemanticReason) {
    items.push('克隆后语义干扰评分尚未生成。')
  }
  if (qualityScore !== null) {
    const qualityPrefix = `克隆音频质量退化评分的值为 ${qualityScore.toFixed(2)} 分，说明听感质量保护效果${higherIsBetterLevel(qualityScore, 70, 85)}`
    if (identityScore !== null && semanticScore !== null) {
      const lowestScore = Math.min(identityScore, semanticScore, qualityScore)
      if (lowestScore >= 85) {
        items.push(`${qualityPrefix}，保护之后在听感、身份、语义综合保护上达到了良好的效果。`)
      } else if (lowestScore >= 70) {
        items.push(`${qualityPrefix}，保护之后在听感、身份、语义综合保护上达到了中等效果。`)
      } else {
        items.push(`${qualityPrefix}，保护之后在听感、身份、语义综合保护上的效果较差。`)
      }
    } else {
      items.push(`${qualityPrefix}，但身份或语义评分尚未完整生成，暂不能给出综合保护结论。`)
    }
  } else {
    items.push('克隆音频质量退化评分尚未生成，身份、语义或听感评分尚未完整生成，暂不能给出综合保护结论。')
  }
  return items
}
