import { useEffect, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { cn } from '@/lib/utils'

export interface MathTextProps {
  /** LaTeX expression, e.g. `L_{\\mathrm{sem}}` */
  formula: string
  /** Whether to render as inline math (default true) */
  inline?: boolean
  className?: string
}

export function MathText({ formula, inline = true, className }: MathTextProps) {
  const ref = useRef<HTMLSpanElement | null>(null)

  useEffect(() => {
    if (!ref.current) return
    katex.render(formula, ref.current, {
      throwOnError: false,
      displayMode: !inline,
    })
  }, [formula, inline])

  return <span ref={ref} className={cn('inline-block', className)} />
}

export function MathBlock({ formula, className }: { formula: string; className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!ref.current) return
    katex.render(formula, ref.current, {
      throwOnError: false,
      displayMode: true,
    })
  }, [formula])

  return <div ref={ref} className={cn('overflow-x-auto py-1', className)} />
}