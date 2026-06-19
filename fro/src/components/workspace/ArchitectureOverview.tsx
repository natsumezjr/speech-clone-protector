import { ArrowDown, ArrowRight, BrainCircuit, Cpu, Ear, Fingerprint, ShieldCheck } from 'lucide-react'
import type { ReactNode } from 'react'
import { Badge } from '@/components/common/Badge'
import { Panel } from '@/components/common/Panel'

export function ArchitectureOverview() {
  return (
    <Panel className="h-full border-sky-400/24 bg-[#071226]/88">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">系统架构概览</h2>
          <p className="mt-1 text-sm text-slate-400">端到端语音克隆防护流程（E2E-VGuard）</p>
        </div>
        <a className="text-sm text-cyan-300 hover:text-cyan-100" href="/results/mock-task-001">
          查看详情
        </a>
      </div>

      <div className="rounded-xl border border-sky-400/18 bg-[#050a19]/68 p-4">
        <div className="grid items-center gap-3 md:grid-cols-[1fr_auto_1.1fr_auto_1fr]">
          <ArchNode title="输入音频" subtitle={<VariableSymbol name="x" />} />
          <ArrowRight className="mx-auto hidden h-6 w-6 text-slate-500 md:block" />
          <div className="rounded-xl border border-sky-400/35 bg-sky-400/10 p-4 text-center">
            <div className="mx-auto grid h-16 w-16 place-items-center rounded-full border border-cyan-300/40 bg-cyan-300/12">
              <ShieldCheck className="h-8 w-8 text-cyan-100" />
            </div>
            <p className="mt-3 font-bold text-cyan-100">防护优化引擎</p>
          </div>
          <ArrowRight className="mx-auto hidden h-6 w-6 text-slate-500 md:block" />
          <ArchNode title="保护音频" subtitle={<VariableSymbol name="x" prime />} />
        </div>

        <div className="mx-auto my-3 flex w-[70%] justify-around">
          <ArrowDown className="h-6 w-6 text-emerald-300" />
          <ArrowDown className="h-6 w-6 text-sky-300" />
          <ArrowDown className="h-6 w-6 text-violet-300" />
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          <BranchCard
            tone="green"
            icon={<BrainCircuit className="h-5 w-5" />}
            title="语义分支"
            items={['ASR 系统', '多模型语义编码', '表示空间约束', '语义漂移评估']}
          />
          <BranchCard
            tone="blue"
            icon={<Fingerprint className="h-5 w-5" />}
            title="Feature 分支"
            items={['Feature Encoder', 'Speaker Embedding', '声学特征约束', '说话人不可恢复']}
          />
          <BranchCard
            tone="purple"
            icon={<Ear className="h-5 w-5" />}
            title="听感约束"
            items={['心理声学模型', '掩蔽阈值建模', '听感优化', '最小化可感知差异']}
          />
        </div>

        <div className="mt-3 rounded-lg border border-sky-400/20 bg-sky-400/8 p-3 text-center text-xs leading-6 text-slate-200">
          <span className="font-serif italic tracking-normal text-slate-100">
            <span>L</span>
            <span> = λ</span>
            <sub>sem</sub>
            <span>L</span>
            <sub>sem</sub>
            <span> + λ</span>
            <sub>feat</sub>
            <span>L</span>
            <sub>feat</sub>
            <span> + λ</span>
            <sub>psy</sub>
            <span>L</span>
            <sub>psy</sub>
            <span> + λ</span>
            <sub>2</sub>
            <span>∥δ∥</span>
            <sub>2</sub>
          </span>
        </div>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-[1fr_0.9fr]">
        <div className="rounded-xl border border-emerald-400/24 bg-emerald-400/8 p-4">
          <h3 className="font-bold text-emerald-200">当前前端支持 Mock / API 快速切换</h3>
          <p className="mt-2 text-xs leading-5 text-slate-300">可用 Mock 数据快速体验完整流程，后续切换到后端 API 获取真实防护结果。</p>
          <div className="mt-3 inline-flex rounded-full border border-cyan-300/20 bg-[#050a19] p-1">
            <span className="rounded-full bg-cyan-400 px-4 py-1.5 text-xs font-bold text-slate-950">Mock 模式</span>
            <span className="px-4 py-1.5 text-xs text-slate-400">API 模式</span>
          </div>
        </div>
        <div className="rounded-xl border border-white/10 bg-[#050a19]/70 p-4">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="font-bold text-white">API 状态</h3>
            <Badge tone="green">在线</Badge>
          </div>
          {[
            ['/api/files/upload', 'GET', '200'],
            ['/api/tasks/protect', 'POST', '200'],
            ['/api/tasks/{id}', 'GET', '200'],
          ].map(([path, method, code]) => (
            <div key={path} className="flex items-center justify-between border-t border-white/[0.08] py-2 text-xs">
              <span className="text-slate-300">{path}</span>
              <span className="text-slate-500">{method}</span>
              <span className="rounded bg-emerald-400/12 px-2 py-0.5 text-emerald-200">{code}</span>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  )
}

function VariableSymbol({ name, prime }: { name: string; prime?: boolean }) {
  return (
    <span className="font-serif italic tracking-normal text-slate-400">
      {name}
      {prime ? <sup>′</sup> : null}
    </span>
  )
}

function ArchNode({ title, subtitle }: { title: string; subtitle: ReactNode }) {
  return (
    <div className="rounded-xl border border-white/10 bg-[#071226] p-4 text-center">
      <Cpu className="mx-auto mb-2 h-8 w-8 text-sky-200" />
      <p className="font-semibold text-white">{title}</p>
      <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
    </div>
  )
}

function BranchCard({
  tone,
  icon,
  title,
  items,
}: {
  tone: 'green' | 'blue' | 'purple'
  icon: ReactNode
  title: string
  items: string[]
}) {
  const classes = {
    green: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-200',
    blue: 'border-sky-400/30 bg-sky-400/10 text-sky-200',
    purple: 'border-violet-400/30 bg-violet-400/10 text-violet-200',
  }[tone]

  return (
    <div className={`rounded-xl border p-3 ${classes}`}>
      <div className="mb-3 flex items-center gap-2 font-bold">
        {icon}
        {title}
      </div>
      <div className="space-y-2">
        {items.map((item) => (
          <div key={item} className="rounded-lg border border-white/10 bg-black/[0.12] px-2 py-2 text-center text-xs text-slate-200">
            {item}
          </div>
        ))}
      </div>
    </div>
  )
}
