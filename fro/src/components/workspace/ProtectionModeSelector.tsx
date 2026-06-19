import { Badge } from '@/components/common/Badge'
import { cn } from '@/lib/utils'

const protectionModes = [
  { value: 'standard', title: '标准保护', description: '平衡安全与听感' },
  { value: 'strong', title: '强保护', description: '更强安全性，略降听感' },
  { value: 'high_fidelity', title: '高保真', description: '更优听感，安全性适中' },
  { value: 'custom', title: '高级自定义', description: '自由调整参数' },
] as const

interface ProtectionModeSelectorProps {
  value: string
  onChange: (value: string) => void
}

export function ProtectionModeSelector({ value, onChange }: ProtectionModeSelectorProps) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {protectionModes.map((mode) => (
        <button
          type="button"
          key={mode.value}
          onClick={() => onChange(mode.value)}
          className={cn(
            'rounded-xl border p-4 text-left transition hover:border-cyan-300/50 hover:bg-cyan-300/8',
            value === mode.value ? 'border-cyan-300/60 bg-cyan-300/10' : 'border-white/10 bg-slate-950/35',
          )}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-semibold text-white">{mode.title}</span>
            {mode.value === 'standard' ? <Badge tone="cyan">默认</Badge> : null}
          </div>
          <p className="mt-2 text-xs text-slate-400">{mode.description}</p>
        </button>
      ))}
    </div>
  )
}
