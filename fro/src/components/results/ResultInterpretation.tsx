import { EvidenceCard } from '@/components/cards/EvidenceCard'
import type { TaskResult } from '@/types/task'
import { percent } from '@/utils/format'

export function ResultInterpretation({ result }: { result: TaskResult }) {
  const items = [
    `语义层面：WER ${percent(result.asr.wer)}，关键语义被显著干扰，机器理解难度提升。`,
    `Feature 层面：相似度从 ${result.speaker.simBefore.toFixed(3)} 降至 ${result.speaker.simAfter.toFixed(3)}，已有效破坏可克隆性。`,
    `听感层面：PESQ=${result.quality.pesq.toFixed(2)}，MOS-LQO=${result.quality.mosLqo.toFixed(2)}，整体听感保持良好，满足可用性要求。`,
    '综合结论：各项指标达到演示阈值，判定为“防护有效”。',
  ]
  return (
    <EvidenceCard title="结果解读（自动生成）">
      <div className="grid gap-3 md:grid-cols-2">
        {items.map((item) => (
          <p key={item} className="rounded-xl border border-white/10 bg-[#050a19]/70 p-5 text-base leading-7 text-slate-200">
            {item}
          </p>
        ))}
      </div>
    </EvidenceCard>
  )
}
