import { Badge } from '@/components/common/Badge'
import { EvidenceCard } from '@/components/cards/EvidenceCard'
import type { AsrMetrics } from '@/types/task'

const protectedSegments = [
  { text: '今天', tone: 'same' },
  { text: '石头很蓝', tone: 'replace' },
  { text: '，我们一路去', tone: 'same' },
  { text: '公元散不唬', tone: 'replace' },
  { text: '。', tone: 'same' },
  { text: '船长胡边走', tone: 'replace' },
  { text: '，你可以买到很多', tone: 'same' },
  { text: '漂多的画', tone: 'replace' },
  { text: '，未分叫过来，甘觉非等似醒。', tone: 'replace' },
  { text: '我们转个地放坐下，聊聊最没的生高和工件，放松一下先青。', tone: 'replace' },
] as const

export function AsrDiffPanel({ asr }: { asr: AsrMetrics }) {
  const editStats = getTextEditStats(asr.originalText, asr.protectedText)
  const cer = asr.cer ?? editStats.cer
  const tokenErrorRate = asr.tokenErrorRate ?? asr.tokenChangeRate
  const insertRate = asr.insertRate ?? editStats.insertRate
  const deleteRate = asr.deleteRate ?? editStats.deleteRate

  return (
    <EvidenceCard title="机器理解分析：ASR 转写对比">
      <div className="grid gap-4 lg:grid-cols-[1fr_280px_1fr]">
        <div className="rounded-xl border border-white/10 bg-[#050a19]/70 p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">原始转写</h3>
            <Badge tone="green">清晰语义</Badge>
          </div>
          <p className="text-base leading-8 text-slate-200">{asr.originalText}</p>
        </div>
        <div className="grid content-center gap-3">
          {[
            ['WER（词错率）', formatOptionalPercent(asr.wer)],
            ['CER（字错率）', formatOptionalPercent(cer)],
            ['Token 错误率', formatOptionalPercent(tokenErrorRate)],
            ['SD（语义漂移）', formatOptionalPercent(asr.semanticDrift)],
            ['IR（插入率）', formatOptionalPercent(insertRate)],
            ['DR（删除率）', formatOptionalPercent(deleteRate)],
          ].map(([label, value]) => (
            <div key={label} className="rounded-xl border border-cyan-300/20 bg-cyan-300/10 p-4 text-center">
              <p className="text-sm text-slate-400">{label}</p>
              <p className="mt-1 text-2xl font-black text-cyan-100">{value}</p>
            </div>
          ))}
        </div>
        <div className="rounded-xl border border-white/10 bg-[#050a19]/70 p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">保护后转写</h3>
            <Badge tone="orange">语义漂移</Badge>
          </div>
          <p className="text-base leading-8 text-slate-200">
            {protectedSegments.map((segment, index) => (
              <span
                key={`${segment.text}-${index}`}
                className={
                  segment.tone === 'replace'
                    ? 'rounded bg-amber-400/15 px-1 text-amber-100 ring-1 ring-amber-300/20'
                    : 'text-slate-300'
                }
              >
                {segment.text}
              </span>
            ))}
          </p>
          <div className="mt-4 flex flex-wrap gap-2 text-xs">
            <span className="rounded bg-emerald-400/15 px-2 py-1 text-emerald-100">新增</span>
            <span className="rounded bg-red-400/15 px-2 py-1 text-red-100">删除</span>
            <span className="rounded bg-amber-400/15 px-2 py-1 text-amber-100">替换</span>
          </div>
        </div>
      </div>
    </EvidenceCard>
  )
}

function formatOptionalPercent(value: unknown) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numberValue)) return '无'
  return `${(numberValue <= 1 ? numberValue * 100 : numberValue).toFixed(1)}%`
}

function getTextEditStats(original: string, next: string) {
  const a = Array.from(original)
  const b = Array.from(next)
  if (a.length === 0) {
    return {
      cer: undefined,
      insertRate: undefined,
      deleteRate: undefined,
    }
  }

  const dp = Array.from({ length: a.length + 1 }, () => Array<number>(b.length + 1).fill(0))

  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }

  let i = 0
  let j = 0
  let insertions = 0
  let deletions = 0

  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      i += 1
      j += 1
    } else if (j < b.length && (i === a.length || dp[i][j + 1] >= dp[i + 1]?.[j])) {
      insertions += 1
      j += 1
    } else if (i < a.length) {
      deletions += 1
      i += 1
    }
  }

  const base = a.length

  return {
    cer: (insertions + deletions) / base,
    insertRate: insertions / base,
    deleteRate: deletions / base,
  }
}
