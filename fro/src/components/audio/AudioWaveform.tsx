import { cn } from '@/lib/utils'

interface AudioWaveformProps {
  dense?: boolean
  variant?: 'cyan' | 'green' | 'orange'
}

export function AudioWaveform({ dense = false, variant = 'cyan' }: AudioWaveformProps) {
  const bars = dense ? 72 : 46
  const color =
    variant === 'green'
      ? 'from-emerald-300 to-cyan-300'
      : variant === 'orange'
        ? 'from-amber-300 to-orange-400'
        : 'from-cyan-200 to-blue-400'

  return (
    <div className="relative h-24 overflow-hidden rounded-xl border border-white/10 bg-slate-950/70 px-3 py-4">
      <div className="absolute inset-x-0 top-1/2 h-px bg-cyan-200/15" />
      <div className="flex h-full items-center gap-1">
        {Array.from({ length: bars }, (_, index) => {
          const height = 18 + Math.abs(Math.sin(index * 0.67) * 42) + Math.abs(Math.cos(index * 0.23) * 18)
          return (
            <span
              key={index}
              className={cn('w-full rounded-full bg-gradient-to-t opacity-80', color)}
              style={{ height: `${Math.min(76, height)}%` }}
            />
          )
        })}
      </div>
    </div>
  )
}
