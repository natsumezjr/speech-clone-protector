import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

type BadgeTone = 'cyan' | 'green' | 'blue' | 'purple' | 'orange' | 'red' | 'slate'

const tones: Record<BadgeTone, string> = {
  cyan: 'border-cyan-400/30 bg-cyan-400/10 text-cyan-100',
  green: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-100',
  blue: 'border-blue-400/30 bg-blue-400/10 text-blue-100',
  purple: 'border-violet-400/30 bg-violet-400/10 text-violet-100',
  orange: 'border-amber-400/30 bg-amber-400/10 text-amber-100',
  red: 'border-red-400/30 bg-red-400/10 text-red-100',
  slate: 'border-slate-500/30 bg-slate-700/40 text-slate-200',
}

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone
}

export function Badge({ className, tone = 'cyan', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium',
        tones[tone],
        className,
      )}
      {...props}
    />
  )
}
