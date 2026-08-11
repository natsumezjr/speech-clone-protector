import { formatAsrRatePercent, generateAsrMetricInsights } from './asrInsightDisplay.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

const chinese = generateAsrMetricInsights({
  metricLevel: 'char',
  wer: 0.72,
  cer: 0.9,
  substituteRate: 0.4,
  deleteRate: 0.2,
  insertRate: 0.3,
  editCounts: {
    level: 'char',
    referenceLength: 10,
    substitutions: 4,
    deletions: 2,
    insertions: 3,
    totalErrors: 9,
  },
})
assert(chinese.includes('WER 的值为 72.00%，CER 的值为 90.00%，说明 ASR 干扰效果优秀。'), 'WER and CER should share one concrete conclusion when their levels match')
assert(chinese.includes('替换率的值为 40.00%，删除率的值为 20.00%，插入率的值为 30.00%，说明主要错误类型为替换。'), 'Error-rate insight should contain concrete values and the dominant error type')
assert(!chinese.join(' ').includes('参考字符'), 'ASR insight should not explain the reference-count calculation')
assert(!chinese.join(' ').includes('后端统计'), 'ASR insight should not explain metric provenance or calculation details')

const lowEvidence = generateAsrMetricInsights({ metricLevel: 'word', wer: 0.19 }).join(' ')
const mediumEvidence = generateAsrMetricInsights({ metricLevel: 'word', wer: 0.2 }).join(' ')
const excellentEvidence = generateAsrMetricInsights({ metricLevel: 'word', wer: 0.5 }).join(' ')
assert(lowEvidence.includes('说明 ASR 干扰效果较差'), 'WER below 20 percent should be rated poor')
assert(mediumEvidence.includes('说明 ASR 干扰效果中等'), 'WER from 20 percent should be rated medium')
assert(excellentEvidence.includes('说明 ASR 干扰效果优秀'), 'WER from 50 percent should be rated excellent')

const overOne = generateAsrMetricInsights({ metricLevel: 'word', wer: 1.25 })
assert(overOne.some((item) => item.includes('WER 的值为 125.00%')), 'WER ratios above one must remain above 100 percent')
assert(!overOne.some((item) => item.includes('允许超过')), 'ASR insight should not explain the calculation rule')
assert(formatAsrRatePercent(1.1614) === '116.14%', 'ASR rate formatting must always treat backend values as ratios')

const withSharedSemantic = generateAsrMetricInsights(
  { metricLevel: 'word', wer: 5 / 12, cer: 0.3051 },
  null,
  {
    tokenChangeRate: 0.9223,
    semanticDrift: 0.58,
    tokenScore: 90.55,
    driftScore: 89.13,
    protectionSemanticScore: 89.91,
  },
)
const semanticText = withSharedSemantic.join(' ')
assert(
  withSharedSemantic.includes('Token 变化率的值为 92.23%，Token 子分的值为 90.55 分，说明离散防护效果优秀。'),
  'Shared semantic raw values should form their own sentence',
)
assert(
  withSharedSemantic.includes('语义表示漂移的值为 0.58，语义漂移子分的值为 89.13 分，ASR 语义保护分的值为 89.91 分，说明语义保护效果优秀。'),
  'Shared semantic scores should form a second sentence with concrete conclusions',
)
assert(!semanticText.includes('计算后'), 'Semantic insight should not describe the score calculation')
assert(!semanticText.includes('加权'), 'Semantic insight should not explain the scoring method')
assert(!semanticText.includes('只计算一次'), 'Semantic insight should not explain metric reuse')

const tokenEditFallback = generateAsrMetricInsights(
  { metricLevel: 'char' },
  null,
  { tokenErrorRate: 0.8655, semanticDrift: 0.55, protectionSemanticScore: 84.99 },
).join(' ')
assert(tokenEditFallback.includes('Token 编辑率的值为 86.55%'), 'Token edit fallback must use its own metric name')
assert(!tokenEditFallback.includes('Token 变化率的值为 86.55%'), 'Token edit rate must not be mislabeled as Token change rate')
assert(tokenEditFallback.includes('说明语义保护效果中等'), 'ASR semantic score below 85 should be rated medium')

const poorSemantic = generateAsrMetricInsights({}, null, { protectionSemanticScore: 69.99 }).join(' ')
const excellentSemantic = generateAsrMetricInsights({}, null, { protectionSemanticScore: 85 }).join(' ')
assert(poorSemantic.includes('说明语义保护效果较差'), 'ASR semantic score below 70 should be rated poor')
assert(excellentSemantic.includes('说明语义保护效果优秀'), 'ASR semantic score from 85 should be rated excellent')

const subscoreLevels = generateAsrMetricInsights({}, null, { tokenScore: 49.99, driftScore: 50, protectionSemanticScore: 70 }).join(' ')
assert(subscoreLevels.includes('Token 子分的值为 49.99 分，说明离散防护效果较差'), 'Token subscore below 50 should be rated poor')
assert(subscoreLevels.includes('语义漂移子分的值为 50.00 分，说明语义漂移防护效果中等'), 'Drift subscore from 50 should be rated medium')

const unavailable = generateAsrMetricInsights({ metricLevel: 'char' })
assert(unavailable.includes('CER 尚未生成。'), 'Missing ASR metrics must stay visibly unavailable')
assert(unavailable.includes('Token 指标与语义表示漂移尚未生成。'), 'Missing shared semantic metrics must stay visibly unavailable')
