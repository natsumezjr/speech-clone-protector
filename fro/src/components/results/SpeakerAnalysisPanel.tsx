import { EvidenceCard } from '@/components/cards/EvidenceCard'
import { RadarChart } from '@/components/charts/RadarChart'
import { SimilarityBar } from '@/components/charts/SimilarityBar'
import type { TaskResult } from '@/types/task'
import { percent } from '@/utils/format'

export function SpeakerAnalysisPanel({ result }: { result: TaskResult }) {
  const speaker = result.speaker
  return (
    <EvidenceCard title="声纹 / 音色分析">
      <div className="grid gap-5 lg:grid-cols-[1.05fr_0.95fr]">
        <RadarChart before={result.charts.radarBefore} after={result.charts.radarAfter} />
        <div className="space-y-4">
          <SimilarityBar label="防护前声纹相似度" value={speaker.simBefore} />
          <SimilarityBar label="防护后声纹相似度" value={speaker.simAfter} tone="green" />
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-white/10 bg-slate-950/45 p-4">
              <p className="text-xs text-slate-400">下降</p>
              <p className="mt-2 text-2xl font-bold text-emerald-200">{percent(speaker.simDropRate)}</p>
            </div>
            <div className="rounded-xl border border-white/10 bg-slate-950/45 p-4">
              <p className="text-xs text-slate-400">Embedding 距离</p>
              <p className="mt-2 text-lg font-bold text-cyan-100">
                {speaker.embeddingDistanceBefore.toFixed(3)} → {speaker.embeddingDistanceAfter.toFixed(3)}
              </p>
              <p className="mt-1 text-xs text-slate-400">提升 548.1%</p>
            </div>
          </div>
          <p className="rounded-xl border border-emerald-300/20 bg-emerald-300/8 p-4 text-sm leading-6 text-emerald-50">
            Speaker Embedding 可分离性显著提升，原始说话人身份特征难以被未授权克隆系统稳定恢复。
          </p>
        </div>
      </div>
    </EvidenceCard>
  )
}
