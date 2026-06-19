import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export function Panel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'rounded-xl border border-sky-400/18 bg-[#0b1425]/86 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03),0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur',
        className,
      )}
      {...props}
    />
  )
}
