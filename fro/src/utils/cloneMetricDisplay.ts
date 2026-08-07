export type CloneMetricInput = {
  originalSimilarity?: number | null
  protectedSimilarity?: number | null
  embeddingDistanceBefore?: number | null
  embeddingDistanceAfter?: number | null
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
  const embeddingDistanceDelta = computeAbsoluteDelta(input.embeddingDistanceBefore, input.embeddingDistanceAfter)
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
  const similarityDropAbs = computeAbsoluteDrop(input.originalSimilarity, input.protectedSimilarity)
  const embeddingDistanceDelta = computeAbsoluteDelta(input.embeddingDistanceBefore, input.embeddingDistanceAfter)
  const items: string[] = []
  if (similarityDropAbs !== null) {
    const direction = similarityDropAbs > 0 ? '绝对下降' : similarityDropAbs < 0 ? '绝对上升' : '无明显变化'
    const magnitude = similarityDropAbs === 0 ? '' : ` ${Math.abs(similarityDropAbs).toFixed(2)}`
    items.push(
      `保护后克隆相似度从 ${formatCloneMetricNumber(input.originalSimilarity)} 变为 ${formatCloneMetricNumber(input.protectedSimilarity)}，${direction}${magnitude}。`,
    )
  }
  if (embeddingDistanceDelta !== null) {
    const direction = embeddingDistanceDelta > 0 ? '增加' : embeddingDistanceDelta < 0 ? '减少' : '无明显变化'
    const magnitude = embeddingDistanceDelta === 0 ? '' : `，绝对${direction} ${Math.abs(embeddingDistanceDelta).toFixed(2)}`
    items.push(
      `embedding cosine distance 从 ${formatCloneMetricNumber(input.embeddingDistanceBefore)} ${direction === '增加' ? '增加到' : direction === '减少' ? '减少到' : '保持在'} ${formatCloneMetricNumber(input.embeddingDistanceAfter)}${magnitude}。`,
    )
    items.push('cosine distance = 1 - cosine similarity，因此当相似度为负时，距离可以大于 1。')
  }
  if ([input.originalSimilarity, input.protectedSimilarity, input.embeddingDistanceBefore, input.embeddingDistanceAfter].some((value) => optionalMetricNumber(value) === null)) {
    items.push('部分克隆指标未生成，不做强结论。')
  }
  if (items.length === 0) items.push('语音克隆评估已执行，但暂时没有足够指标用于生成结论。')
  return items
}
