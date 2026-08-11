export type AsrInsightEditCounts = {
  level?: string | null
  referenceLength?: number | null
  substitutions?: number | null
  insertions?: number | null
  deletions?: number | null
  totalErrors?: number | null
} | null

export type AsrInsightErrorShares = {
  substituteShare?: number | null
  insertShare?: number | null
  deleteShare?: number | null
} | null

export type AsrInsightInput = {
  wer?: number | null
  cer?: number | null
  substituteRate?: number | null
  insertRate?: number | null
  deleteRate?: number | null
  editCounts?: AsrInsightEditCounts
  errorShares?: AsrInsightErrorShares
  metricLevel?: string | null
}

export type AsrInsightFallback = {
  level: 'word' | 'char'
  werOrCer: number
  substituteRate: number
  insertRate: number
  deleteRate: number
  referenceLength: number
  substitutions: number
  insertions: number
  deletions: number
  totalErrors: number
  errorShares: {
    substituteShare: number
    insertShare: number
    deleteShare: number
  }
} | null

export type SharedSemanticInsightInput = {
  tokenChangeRate?: number | null
  tokenErrorRate?: number | null
  semanticDrift?: number | null
  tokenScore?: number | null
  driftScore?: number | null
  protectionSemanticScore?: number | null
  semanticIsMfccProxy?: boolean
} | null

function finiteNumber(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function firstNumber(...values: unknown[]) {
  for (const value of values) {
    const numberValue = finiteNumber(value)
    if (numberValue !== null) return numberValue
  }
  return null
}

export function formatAsrRatePercent(value: unknown) {
  const numberValue = finiteNumber(value)
  return numberValue === null ? '未生成' : `${(numberValue * 100).toFixed(2)}%`
}

function completeEditCounts(value: AsrInsightEditCounts | undefined) {
  if (!value) return null
  const referenceLength = finiteNumber(value.referenceLength)
  const substitutions = finiteNumber(value.substitutions)
  const insertions = finiteNumber(value.insertions)
  const deletions = finiteNumber(value.deletions)
  const totalErrors = finiteNumber(value.totalErrors)
  if ([referenceLength, substitutions, insertions, deletions, totalErrors].some((item) => item === null)) return null
  return {
    referenceLength: referenceLength as number,
    substitutions: substitutions as number,
    insertions: insertions as number,
    deletions: deletions as number,
    totalErrors: totalErrors as number,
  }
}

function fallbackEditCounts(value: AsrInsightFallback) {
  if (!value) return null
  return {
    referenceLength: value.referenceLength,
    substitutions: value.substitutions,
    insertions: value.insertions,
    deletions: value.deletions,
    totalErrors: value.totalErrors,
  }
}

function higherIsBetterLevel(value: number, mediumThreshold: number, excellentThreshold: number) {
  if (value >= excellentThreshold) return '优秀'
  if (value >= mediumThreshold) return '中等'
  return '较差'
}

function asrEvidenceLevel(rate: number) {
  return higherIsBetterLevel(rate, 0.2, 0.5)
}

function appendSharedSemanticInsights(items: string[], semantic: SharedSemanticInsightInput) {
  if (!semantic) {
    items.push('Token 指标与语义表示漂移尚未生成。')
    return
  }

  const tokenChangeRate = finiteNumber(semantic.tokenChangeRate)
  const tokenErrorRate = finiteNumber(semantic.tokenErrorRate)
  const semanticDrift = finiteNumber(semantic.semanticDrift)
  const tokenScore = finiteNumber(semantic.tokenScore)
  const driftScore = finiteNumber(semantic.driftScore)
  const protectionSemanticScore = finiteNumber(semantic.protectionSemanticScore)
  const tokenMetricName = tokenChangeRate !== null ? 'Token 变化率' : tokenErrorRate !== null ? 'Token 编辑率' : null
  const tokenMetricValue = tokenChangeRate ?? tokenErrorRate
  const driftMetricName = semantic.semanticIsMfccProxy ? 'MFCC 代理漂移' : '语义表示漂移'

  // Token 句：变化率/编辑率 + Token 子分
  const tokenParts: string[] = []
  if (tokenMetricName && tokenMetricValue !== null) tokenParts.push(`${tokenMetricName}的值为 ${formatAsrRatePercent(tokenMetricValue)}`)
  if (tokenScore !== null) tokenParts.push(`Token 子分的值为 ${tokenScore.toFixed(2)} 分，说明离散防护效果${higherIsBetterLevel(tokenScore, 50, 80)}`)

  // 语义句：语义表示漂移 + 语义漂移子分 + ASR 语义保护分
  const semanticParts: string[] = []
  if (semanticDrift !== null) semanticParts.push(`${driftMetricName}的值为 ${semanticDrift.toFixed(2)}`)
  if (driftScore !== null) semanticParts.push(`语义漂移子分的值为 ${driftScore.toFixed(2)} 分`)
  if (protectionSemanticScore !== null) semanticParts.push(`ASR 语义保护分的值为 ${protectionSemanticScore.toFixed(2)} 分，说明语义保护效果${higherIsBetterLevel(protectionSemanticScore, 70, 85)}`)

  if (tokenParts.length > 0) items.push(`${tokenParts.join('，')}。`)
  if (semanticParts.length > 0) items.push(`${semanticParts.join('，')}。`)
  if (tokenParts.length === 0 && semanticParts.length === 0) {
    items.push('Token 指标与语义表示漂移尚未生成。')
  }
}

export function generateAsrMetricInsights(
  input: AsrInsightInput,
  fallback: AsrInsightFallback = null,
  semantic: SharedSemanticInsightInput = null,
) {
  const explicitLevel = String(input.metricLevel ?? input.editCounts?.level ?? fallback?.level ?? '').toLowerCase()
  const level = explicitLevel === 'char' || explicitLevel === 'word'
    ? explicitLevel
    : finiteNumber(input.cer) !== null && finiteNumber(input.wer) === null ? 'char' : 'word'
  const metricName = level === 'char' ? 'CER' : 'WER'
  const werRate = firstNumber(input.wer, fallback?.level === 'word' ? fallback.werOrCer : null)
  const cerRate = firstNumber(input.cer, fallback?.level === 'char' ? fallback.werOrCer : null)
  const counts = completeEditCounts(input.editCounts) ?? fallbackEditCounts(fallback)
  const rateFromCount = (count: number | undefined) => (
    counts && count !== undefined ? count / Math.max(counts.referenceLength, 1) : null
  )
  const rates = {
    substitutions: firstNumber(input.substituteRate, rateFromCount(counts?.substitutions), fallback?.substituteRate),
    insertions: firstNumber(input.insertRate, rateFromCount(counts?.insertions), fallback?.insertRate),
    deletions: firstNumber(input.deleteRate, rateFromCount(counts?.deletions), fallback?.deleteRate),
  }
  const items: string[] = []

  if (werRate !== null && cerRate !== null) {
    const werLevel = asrEvidenceLevel(werRate)
    const cerLevel = asrEvidenceLevel(cerRate)
    items.push(werLevel === cerLevel
      ? `WER 的值为 ${formatAsrRatePercent(werRate)}，CER 的值为 ${formatAsrRatePercent(cerRate)}，说明 ASR 干扰效果${werLevel}。`
      : `WER 的值为 ${formatAsrRatePercent(werRate)}，说明 ASR 干扰效果${werLevel}；CER 的值为 ${formatAsrRatePercent(cerRate)}，说明 ASR 干扰效果${cerLevel}。`)
  } else if (werRate !== null) {
    items.push(`WER 的值为 ${formatAsrRatePercent(werRate)}，说明 ASR 干扰效果${asrEvidenceLevel(werRate)}。`)
  } else if (cerRate !== null) {
    items.push(`CER 的值为 ${formatAsrRatePercent(cerRate)}，说明 ASR 干扰效果${asrEvidenceLevel(cerRate)}。`)
  } else {
    items.push(`${metricName} 尚未生成。`)
  }

  const errorRates = [
    { label: '替换率', value: rates.substitutions },
    { label: '删除率', value: rates.deletions },
    { label: '插入率', value: rates.insertions },
  ].filter((item): item is { label: string; value: number } => item.value !== null)
  if (errorRates.length > 0) {
    const maximum = Math.max(...errorRates.map((item) => item.value))
    const dominant = errorRates.filter((item) => item.value === maximum).map((item) => item.label.replace('率', '')).join('、')
    items.push(`${errorRates.map((item) => `${item.label}的值为 ${formatAsrRatePercent(item.value)}`).join('，')}，说明主要错误类型为${dominant}。`)
  }

  appendSharedSemanticInsights(items, semantic)
  return items
}
