import { Badge } from '@/components/common/Badge'
import { cn } from '@/lib/utils'

const targets = [
  { value: 'semantic', label: '语义防护', description: '干扰 ASR / Tokenizer / LALM 理解' },
  { value: 'timbre', label: '音色防护', description: '降低 Speaker Embedding 可建模性' },
] as const

interface ProtectionTargetSelectorProps {
  value: string[]
  onChange: (value: string[]) => void
}

export function ProtectionTargetSelector({ value, onChange }: ProtectionTargetSelectorProps) {
  const toggle = (target: string) => {
    onChange(value.includes(target) ? value.filter((item) => item !== target) : [...value, target])
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {targets.map((target) => {
        const selected = value.includes(target.value)
        return (
          <button
            type="button"
            key={target.value}
            onClick={() => toggle(target.value)}
            className={cn(
              'rounded-xl border p-4 text-left transition hover:border-emerald-300/50 hover:bg-emerald-300/8',
              selected ? 'border-emerald-300/60 bg-emerald-300/10' : 'border-white/10 bg-slate-950/35',
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-semibold text-white">{target.label}</span>
              {selected ? <Badge tone="green">已启用</Badge> : null}
            </div>
            <p className="mt-2 text-xs text-slate-400">{target.description}</p>
          </button>
        )
      })}
      <button
        type="button"
        onClick={() => onChange(['semantic', 'timbre'])}
        className={cn(
          'rounded-xl border p-4 text-left transition sm:col-span-2',
          value.length === 2 ? 'border-cyan-300/60 bg-cyan-300/10' : 'border-white/10 bg-slate-950/35 hover:bg-cyan-300/8',
        )}
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-white">联合防护</span>
          <Badge tone="cyan">推荐</Badge>
        </div>
        <p className="mt-2 text-xs text-slate-400">语义分支与音色分支联合优化，适合评委演示主流程。</p>
      </button>
    </div>
  )
}
