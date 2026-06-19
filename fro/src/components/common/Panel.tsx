import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export function Panel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'rounded-2xl border border-cyan-400/15 bg-slate-900/62 p-5 shadow-[0_0_40px_rgba(6,182,212,0.08)] backdrop-blur',
        className,
      )}
      {...props}
    />
  )
}
