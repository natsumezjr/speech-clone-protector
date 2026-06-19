import { ArrowRight, BrainCircuit, Ear, Fingerprint, RadioTower } from 'lucide-react'
import { Badge } from '@/components/common/Badge'
import { Panel } from '@/components/common/Panel'

export function ArchitectureOverview() {
  const branches = [
    {
      title: '语义分支',
      icon: <BrainCircuit className="h-5 w-5" />,
      items: ['ASR 系统', '多模型语义编码', 'S3 / HuBERT / Whisper / MFCC', '表示空间约束', '语义漂移评估'],
    },
    {
      title: '音色分支',
      icon: <Fingerprint className="h-5 w-5" />,
      items: ['Timbre Encoder', 'Speaker Embedding', '声纹特征约束', '说话人不可恢复'],
    },
    {
      title: '听感约束',
      icon: <Ear className="h-5 w-5" />,
      items: ['心理声学模型', '掩蔽阈值建模', '听感优化', '最小化可感知差异'],
    },
  ]

  return (
    <Panel>
      <h2 className="mb-5 text-lg font-semibold text-white">系统架构概览</h2>
      <div className="grid items-center gap-3 text-center text-sm">
        <div className="rounded-xl border border-white/10 bg-slate-950/50 p-3">输入音频 x</div>
        <ArrowRight className="mx-auto h-5 w-5 rotate-90 text-cyan-200" />
        <div className="rounded-xl border border-cyan-300/30 bg-cyan-300/10 p-4">
          <RadioTower className="mx-auto mb-2 h-6 w-6 text-cyan-100" />
          防护优化引擎
        </div>
        <ArrowRight className="mx-auto h-5 w-5 rotate-90 text-cyan-200" />
        <div className="rounded-xl border border-emerald-300/30 bg-emerald-300/10 p-3">保护音频 x'</div>
      </div>

      <div className="mt-5 space-y-3">
        {branches.map((branch) => (
          <div key={branch.title} className="rounded-xl border border-white/10 bg-slate-950/40 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-white">
              <span className="text-cyan-100">{branch.icon}</span>
              {branch.title}
            </div>
            <div className="flex flex-wrap gap-2">
              {branch.items.map((item) => (
                <Badge key={item} tone="slate">
                  {item}
                </Badge>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-5 rounded-xl border border-violet-300/20 bg-violet-300/8 p-4 font-mono text-xs leading-6 text-violet-100">
        L = λ_sem L_sem + λ_timbre L_timbre + λ_psy L_psy + λ_2 ||δ||_2
      </div>
    </Panel>
  )
}
