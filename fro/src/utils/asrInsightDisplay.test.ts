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
assert(chinese[0].includes('10 个参考字符、9 次编辑错误'), 'ASR insight should include concrete reference and error counts')
assert(chinese[0].includes('CER 为 90.00%'), 'ASR insight should include the concrete primary CER')
assert(chinese[0].includes('辅助 WER 为 72.00%'), 'ASR insight should identify the auxiliary metric')
assert(chinese[1].includes('替换 4 次（SR 40.00%，占全部错误 44.44%）'), 'ASR insight should explain the concrete error composition')
assert(chinese[2].includes('识别偏离显著'), 'ASR insight should provide a numeric-level conclusion')

const overOne = generateAsrMetricInsights({
  metricLevel: 'word',
  wer: 1.25,
  editCounts: {
    level: 'word',
    referenceLength: 4,
    substitutions: 2,
    deletions: 1,
    insertions: 2,
    totalErrors: 5,
  },
})
assert(overOne.some((item) => item.includes('WER 为 125.00%')), 'WER ratios above one must remain above 100 percent')
assert(overOne.some((item) => item.includes('不是百分比溢出')), 'ASR insight should explain why WER/CER can exceed 100 percent')
assert(formatAsrRatePercent(1.1614) === '116.14%', 'ASR rate formatting must always treat backend values as ratios')

const withSharedSemantic = generateAsrMetricInsights(
  {
    metricLevel: 'word',
    wer: 5 / 12,
    cer: 0.3051,
    editCounts: {
      level: 'word',
      referenceLength: 12,
      substitutions: 5,
      deletions: 0,
      insertions: 0,
      totalErrors: 5,
    },
  },
  null,
  {
    tokenChangeRate: 0.8684,
    semanticDrift: 0.5535,
    tokenScore: 88.12,
    driftScore: 87.34,
    protectionSemanticScore: 87.77,
  },
)
const semanticText = withSharedSemantic.join(' ')
assert(semanticText.includes('Token 变化率为 86.84%'), 'Shared Token change interpretation should use the concrete task value')
assert(semanticText.includes('语义表示漂移为 0.55'), 'Shared semantic drift interpretation should keep two decimals')
assert(semanticText.includes('只计算一次，所有 ASR 模型共用'), 'Shared semantic metrics must not be presented as per-model ASR results')
assert(semanticText.includes('Token 子分为 88.12 分'), 'Semantic score interpretation should include the concrete Token subscore')
assert(semanticText.includes('按 55% 与 45% 加权后'), 'Semantic score interpretation should follow the target weighting')
assert(semanticText.includes('87.77 分'), 'Semantic score interpretation should include the concrete combined score')

const tokenEditFallback = generateAsrMetricInsights(
  { metricLevel: 'char' },
  null,
  { tokenErrorRate: 0.8655, semanticDrift: 0.55 },
).join(' ')
assert(tokenEditFallback.includes('Token 编辑率为 86.55%'), 'Token edit fallback must use its own metric name')
assert(!tokenEditFallback.includes('Token 变化率为 86.55%'), 'Token edit rate must not be mislabeled as Token change rate')

const unavailable = generateAsrMetricInsights({ metricLevel: 'char' })
assert(unavailable.some((item) => item.includes('主指标 CER 尚未生成')), 'Missing ASR metrics must stay visibly unavailable')
assert(unavailable.some((item) => item.includes('Token 指标与语义漂移尚未生成')), 'Missing shared semantic metrics must stay visibly unavailable')
