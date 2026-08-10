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

export function generateCloneMetricInsights(input: CloneMetricInput) {
  const embeddingDistanceDelta = optionalMetricNumber(input.embeddingDistanceDelta)
    ?? computeAbsoluteDelta(input.embeddingDistanceBefore, input.embeddingDistanceAfter)
  const identityScore = optionalMetricNumber(input.cloneIdentityScore)
  const semanticScore = optionalMetricNumber(input.cloneSemanticScore)
  const qualityScore = optionalMetricNumber(input.cloneQualityScore)
  const qualityRawScore = optionalMetricNumber(input.cloneQualityRawScore)
  const qualityRelevance = optionalMetricNumber(input.cloneQualityRelevance)
  const qualityDropRate = optionalMetricNumber(input.cloneQualityDropRate)
  const items: string[] = []
  if (embeddingDistanceDelta !== null) {
    const level = embeddingDistanceDelta >= 0.2 ? '明显' : embeddingDistanceDelta >= 0.05 ? '中等' : '偏低'
    items.push(`身份差异${level}：声纹嵌入距离由 ${formatCloneMetricNumber(input.embeddingDistanceBefore)} 变为 ${formatCloneMetricNumber(input.embeddingDistanceAfter)}，克隆后的声音身份${embeddingDistanceDelta > 0 ? '与原说话人进一步分离' : embeddingDistanceDelta < 0 ? '与原说话人更加接近' : '未出现明显变化'}。`)
  }
  if (semanticScore !== null) {
    const level = semanticScore >= 70 ? '较高' : semanticScore >= 40 ? '中等' : '偏低'
    items.push(`语义干扰${level}：克隆后语义干扰评分为 ${semanticScore.toFixed(2)} 分，反映保护前后克隆表达内容的实际变化。`)
  } else if (input.cloneSemanticReason) {
    items.push('语义干扰暂不可用：克隆语音文本尚未完整生成。')
  }
  if (qualityScore !== null) {
    const dropText = qualityDropRate === null ? '' : `，语音质量相对下降 ${(qualityDropRate * 100).toFixed(2)}%`
    if (qualityRawScore !== null && qualityRelevance !== null) {
      const relevanceText = `${(qualityRelevance * 100).toFixed(2)}%`
      const explanation = qualityRelevance < 0.5
        ? '身份与语义保护已较充分，额外降低听感质量的必要性较低'
        : '当前结果仍需较多参考实际语音质量下降'
      items.push(`克隆后语音质量下降评分为 ${qualityScore.toFixed(2)} 分：${explanation}；原始质量退化分为 ${qualityRawScore.toFixed(2)} 分，参考占比为 ${relevanceText}${dropText}。`)
    } else {
      items.push(`克隆后语音质量下降评分为 ${qualityScore.toFixed(2)} 分${dropText}。`)
    }
  } else if (input.cloneQualityReason) {
    items.push('克隆质量退化暂不可用：语音质量评分尚未生成。')
  }
  if (identityScore !== null) {
    const level = identityScore >= 85 ? '优秀' : identityScore >= 70 ? '中等' : '较差'
    items.push(`身份保护效果${level}：克隆身份保护评分为 ${identityScore.toFixed(2)} 分。`)
  }
  if (identityScore !== null && semanticScore !== null && qualityScore !== null) {
    const scores = [identityScore, semanticScore, qualityScore]
    const conclusion = scores.every((score) => score >= 70)
      ? '总体克隆防护效果良好。'
      : scores.filter((score) => score >= 40).length >= 2
        ? '总体克隆防护效果中等。'
        : '总体克隆防护效果偏弱。'
    items.push(conclusion)
  }
  if (items.length === 0) items.push('语音克隆评估已执行，但身份、语义或质量评分尚未生成。')
  return items
}
