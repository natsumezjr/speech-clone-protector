import { EvidenceCard } from '@/components/cards/EvidenceCard'
import { PsychoacousticChart } from '@/components/charts/PsychoacousticChart'
import type { TaskResult } from '@/types/task'

export function QualityPanel({ result }: { result: TaskResult }) {
  const metrics = [
    ['SNR', `${result.quality.snr.toFixed(1)} dB`, '优秀'],
    ['PESQ', result.quality.pesq.toFixed(2), '良好'],
    ['MOS-LQO', `${result.quality.mosLqo.toFixed(2)} / 5`, '良好'],
  ]

  return (
    <EvidenceCard title="感知质量评估">
      <div className="grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
        <div className="grid content-start gap-3">
          {metrics.map(([label, value, level]) => (
            <div key={label} className="rounded-xl border border-white/10 bg-[#050a19]/70 p-5">
              <p className="text-xs text-slate-400">{label}</p>
              <p className="mt-2 text-2xl font-bold text-cyan-100">{value}</p>
              <p className="mt-1 text-xs text-emerald-200">{level}</p>
            </div>
          ))}
          <p className="rounded-xl border border-cyan-300/20 bg-cyan-300/8 p-4 text-sm leading-6 text-cyan-50">
            扰动大多低于掩蔽阈值，保证防护效果的同时维持可接受听感。
          </p>
        </div>
        <PsychoacousticChart data={result.charts.psychoacoustic} />
      </div>
    </EvidenceCard>
  )
}
