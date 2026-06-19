import { cn } from '@/lib/utils'

interface SimilarityBarProps {
  label: string
  value: number
  tone?: 'cyan' | 'green' | 'orange'
}

export function SimilarityBar({ label, value, tone = 'cyan' }: SimilarityBarProps) {
  const color = tone === 'green' ? 'bg-emerald-400' : tone === 'orange' ? 'bg-amber-400' : 'bg-cyan-300'
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-semibold text-white">{value.toFixed(3)}</span>
      </div>
      <div className="h-2 rounded-full bg-slate-800">
        <div className={cn('h-full rounded-full', color)} style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }} />
      </div>
    </div>
  )
}
