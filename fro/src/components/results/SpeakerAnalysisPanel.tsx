import { EvidenceCard } from '@/components/cards/EvidenceCard'
import { RadarChart } from '@/components/charts/RadarChart'
import { SimilarityBar } from '@/components/charts/SimilarityBar'
import type { TaskResult } from '@/types/task'
import { percent } from '@/utils/format'

export function SpeakerAnalysisPanel({ result }: { result: TaskResult }) {
  const speaker = result.speaker
  return (
    <EvidenceCard title="Feature / 声学特征分析">
      <div className="grid gap-5 lg:grid-cols-[1.05fr_0.95fr]">
        <RadarChart before={result.charts.radarBefore} after={result.charts.radarAfter} />
        <div className="space-y-4">
          <SimilarityBar label="防护前 Feature 相似度" value={speaker.simBefore} />
          <SimilarityBar label="防护后 Feature 相似度" value={speaker.simAfter} tone="green" />
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-white/10 bg-[#050a19]/70 p-5">
              <p className="text-xs text-slate-400">下降</p>
              <p className="mt-2 text-2xl font-bold text-emerald-200">{percent(speaker.simDropRate)}</p>
            </div>
            <div className="rounded-xl border border-white/10 bg-[#050a19]/70 p-5">
              <p className="text-xs text-slate-400">Embedding 距离</p>
              <p className="mt-2 text-lg font-bold text-cyan-100">
                {speaker.embeddingDistanceBefore.toFixed(3)} → {speaker.embeddingDistanceAfter.toFixed(3)}
              </p>
              <p className="mt-1 text-xs text-slate-400">提升 548.1%</p>
            </div>
          </div>
          <p className="rounded-xl border border-emerald-300/20 bg-emerald-300/8 p-4 text-sm leading-6 text-emerald-50">
            声学特征可分离性显著提升，原始说话人身份特征难以被未授权克隆系统稳定恢复。
          </p>
        </div>
      </div>
    </EvidenceCard>
  )
}
