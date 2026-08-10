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

function formatCount(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
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

function resolvedShares(direct: AsrInsightErrorShares | undefined, counts: ReturnType<typeof completeEditCounts>, fallback: AsrInsightFallback) {
  const fromCount = (count: number | undefined) => {
    if (!counts || count === undefined) return null
    return counts.totalErrors > 0 ? count / counts.totalErrors : 0
  }
  return {
    substituteShare: firstNumber(direct?.substituteShare, fromCount(counts?.substitutions), fallback?.errorShares.substituteShare),
    insertShare: firstNumber(direct?.insertShare, fromCount(counts?.insertions), fallback?.errorShares.insertShare),
    deleteShare: firstNumber(direct?.deleteShare, fromCount(counts?.deletions), fallback?.errorShares.deleteShare),
  }
}

function errorLevel(rate: number) {
  if (rate === 0) return '未产生识别偏离'
  if (rate < 0.1) return '识别偏离较小'
  if (rate < 0.3) return '已产生一定识别偏离'
  if (rate < 0.6) return '识别偏离较明显'
  return '识别偏离显著'
}

function appendSharedSemanticInsights(items: string[], semantic: SharedSemanticInsightInput) {
  if (!semantic) {
    items.push('保护任务共享语义指标：Token 指标与语义漂移尚未生成。')
    return
  }

  const tokenChangeRate = finiteNumber(semantic.tokenChangeRate)
  const tokenErrorRate = finiteNumber(semantic.tokenErrorRate)
  const semanticDrift = finiteNumber(semantic.semanticDrift)
  const rawDetails: string[] = []
  if (tokenChangeRate !== null) {
    rawDetails.push(`Token 变化率为 ${formatAsrRatePercent(tokenChangeRate)}，表示两侧较短 Token 序列内同位置 Token 不一致的比例`)
  } else if (tokenErrorRate !== null) {
    rawDetails.push(`Token 编辑率为 ${formatAsrRatePercent(tokenErrorRate)}，表示 Token 序列编辑距离相对原始 Token 序列长度的比例`)
  }
  if (semanticDrift !== null) {
    rawDetails.push(semantic.semanticIsMfccProxy
      ? `MFCC 代理漂移为 ${semanticDrift.toFixed(2)}，反映声学特征变化，不等同于深度语义表示`
      : `语义表示漂移为 ${semanticDrift.toFixed(2)}，数值越高表示原音频与保护音频的连续语义表示差异越大`)
  }
  items.push(rawDetails.length
    ? `保护任务共享语义指标：${rawDetails.join('；')}。两项由原音频和保护音频只计算一次，所有 ASR 模型共用。`
    : '保护任务共享语义指标：Token 指标与语义漂移尚未生成。')

  const tokenScore = finiteNumber(semantic.tokenScore)
  const driftScore = finiteNumber(semantic.driftScore)
  const protectionSemanticScore = finiteNumber(semantic.protectionSemanticScore)
  if (tokenScore !== null && driftScore !== null && protectionSemanticScore !== null) {
    items.push(`语义干扰评分：Token 子分为 ${tokenScore.toFixed(2)} 分，语义漂移子分为 ${driftScore.toFixed(2)} 分；按 55% 与 45% 加权后，保护后音频语义干扰评分为 ${protectionSemanticScore.toFixed(2)} 分。分数越高，表示语义保护干扰越强。`)
  } else if (protectionSemanticScore !== null) {
    items.push(`语义干扰评分：保护后音频语义干扰评分为 ${protectionSemanticScore.toFixed(2)} 分；分数越高，表示语义保护干扰越强。`)
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
  const auxiliaryMetricName = level === 'char' ? 'WER' : 'CER'
  const unitName = level === 'char' ? '字符' : '词'
  const primaryRate = level === 'char'
    ? firstNumber(input.cer, fallback?.level === 'char' ? fallback.werOrCer : null)
    : firstNumber(input.wer, fallback?.level === 'word' ? fallback.werOrCer : null)
  const auxiliaryRate = level === 'char' ? finiteNumber(input.wer) : finiteNumber(input.cer)
  const backendCounts = completeEditCounts(input.editCounts)
  const counts = backendCounts ?? fallbackEditCounts(fallback)
  const rateFromCount = (count: number | undefined) => (
    counts && count !== undefined ? count / Math.max(counts.referenceLength, 1) : null
  )
  const rates = {
    substitutions: firstNumber(input.substituteRate, rateFromCount(counts?.substitutions), fallback?.substituteRate),
    insertions: firstNumber(input.insertRate, rateFromCount(counts?.insertions), fallback?.insertRate),
    deletions: firstNumber(input.deleteRate, rateFromCount(counts?.deletions), fallback?.deleteRate),
  }
  const shares = resolvedShares(input.errorShares, counts, fallback)
  const items: string[] = []

  if (primaryRate === null) {
    const auxiliaryText = auxiliaryRate === null ? '' : `；辅助 ${auxiliaryMetricName} 为 ${formatAsrRatePercent(auxiliaryRate)}`
    items.push(`本次按${unitName}统计的主指标 ${metricName} 尚未生成${auxiliaryText}，暂不能完成当前统计层级的数字解读。`)
  } else {
    const auxiliaryText = auxiliaryRate === null ? '' : `；辅助 ${auxiliaryMetricName} 为 ${formatAsrRatePercent(auxiliaryRate)}`
    if (counts) {
      const sourceText = backendCounts ? '后端统计' : '根据页面转写文本估算'
      items.push(`文本错误：本次按${unitName}统计，${sourceText}共有 ${formatCount(counts.referenceLength)} 个参考${unitName}、${formatCount(counts.totalErrors)} 次编辑错误，${metricName} 为 ${formatAsrRatePercent(primaryRate)}${auxiliaryText}。`)
    } else {
      items.push(`文本错误：本次按${unitName}统计，${metricName} 为 ${formatAsrRatePercent(primaryRate)}${auxiliaryText}。`)
    }

    if (counts) {
      const breakdown = [
        { label: '替换', abbreviation: 'SR', count: counts.substitutions, rate: rates.substitutions, share: shares.substituteShare },
        { label: '删除', abbreviation: 'DR', count: counts.deletions, rate: rates.deletions, share: shares.deleteShare },
        { label: '插入', abbreviation: 'IR', count: counts.insertions, rate: rates.insertions, share: shares.insertShare },
      ]
      const maxCount = Math.max(...breakdown.map((item) => item.count))
      const dominant = maxCount > 0 ? breakdown.filter((item) => item.count === maxCount).map((item) => item.label).join('、') : ''
      const details = breakdown.map((item) => {
        const annotations = [
          item.rate === null ? null : `${item.abbreviation} ${formatAsrRatePercent(item.rate)}`,
          item.share === null ? null : `占全部错误 ${formatAsrRatePercent(item.share)}`,
        ].filter((value): value is string => Boolean(value))
        return `${item.label} ${formatCount(item.count)} 次${annotations.length ? `（${annotations.join('，')}）` : ''}`
      }).join('、')
      items.push(`错误构成：${details}；${dominant ? `主要错误类型为${dominant}` : '三类编辑错误均为 0 次'}。`)
    } else if (Object.values(rates).some((value) => value !== null)) {
      const rateDetails = [
        ['替换率 SR', rates.substitutions],
        ['删除率 DR', rates.deletions],
        ['插入率 IR', rates.insertions],
      ].filter((item): item is [string, number] => item[1] !== null)
      const maxRate = Math.max(...rateDetails.map(([, rate]) => rate))
      const dominant = rateDetails.filter(([, rate]) => rate === maxRate).map(([label]) => label.replace(/率\s[A-Z]+$/, '')).join('、')
      items.push(`错误构成：${rateDetails.map(([label, rate]) => `${label} 为 ${formatAsrRatePercent(rate)}`).join('、')}；占参考${unitName}比例最高的是${dominant}。`)
    }

    let conclusion = `识别影响：${metricName} 为 ${formatAsrRatePercent(primaryRate)}，${errorLevel(primaryRate)}。`
    if (primaryRate > 1) {
      conclusion += ` ${metricName} 允许超过 100%，因为插入、删除和替换的编辑错误总数可以多于参考${unitName}数，这不是百分比溢出。`
    }
    items.push(conclusion)
  }

  appendSharedSemanticInsights(items, semantic)
  return items
}
