import { Badge } from '@/components/common/Badge'
import { EvidenceCard } from '@/components/cards/EvidenceCard'
import type { AsrMetrics } from '@/types/task'
import { percent } from '@/utils/format'

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
  return (
    <EvidenceCard title="机器理解分析：ASR 转写对比">
      <div className="grid gap-4 lg:grid-cols-[1fr_240px_1fr]">
        <div className="rounded-xl border border-white/10 bg-slate-950/45 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">原始转写</h3>
            <Badge tone="green">清晰语义</Badge>
          </div>
          <p className="text-sm leading-7 text-slate-200">{asr.originalText}</p>
        </div>
        <div className="grid content-center gap-3">
          {[
            ['WER', percent(asr.wer)],
            ['CER', percent(asr.cer)],
            ['Token 变化率', percent(asr.tokenChangeRate)],
            ['Semantic Drift', asr.semanticDrift.toFixed(2)],
          ].map(([label, value]) => (
            <div key={label} className="rounded-xl border border-cyan-300/20 bg-cyan-300/8 p-3 text-center">
              <p className="text-xs text-slate-400">{label}</p>
              <p className="mt-1 text-lg font-bold text-cyan-100">{value}</p>
            </div>
          ))}
        </div>
        <div className="rounded-xl border border-white/10 bg-slate-950/45 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">保护后转写</h3>
            <Badge tone="orange">语义漂移</Badge>
          </div>
          <p className="text-sm leading-7 text-slate-200">
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
