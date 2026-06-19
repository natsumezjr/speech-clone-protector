import type { ReactNode } from 'react'
import { Badge } from '@/components/common/Badge'

interface PageHeaderProps {
  eyebrow?: string
  title: string
  description?: string
  actions?: ReactNode
  align?: 'left' | 'center'
}

export function PageHeader({ eyebrow, title, description, actions, align = 'left' }: PageHeaderProps) {
  const centered = align === 'center'
  return (
    <div className={`mb-7 flex flex-col justify-between gap-4 ${centered ? 'items-center text-center' : 'lg:flex-row lg:items-end'}`}>
      <div className={centered ? 'mx-auto max-w-4xl' : ''}>
        {eyebrow ? (
          <Badge tone="cyan" className="mb-4 h-8 px-4 text-sm">
            {eyebrow}
          </Badge>
        ) : null}
        <h1 className="text-[34px] font-bold leading-tight tracking-normal text-white md:text-[44px]">{title}</h1>
        {description ? <p className={`mt-3 max-w-3xl text-base leading-7 text-slate-300 ${centered ? 'mx-auto' : ''}`}>{description}</p> : null}
      </div>
      {actions}
    </div>
  )
}
