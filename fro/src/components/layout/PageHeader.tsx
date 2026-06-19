import type { ReactNode } from 'react'
import { Badge } from '@/components/common/Badge'

interface PageHeaderProps {
  eyebrow?: string
  title: string
  description?: string
  actions?: ReactNode
}

export function PageHeader({ eyebrow, title, description, actions }: PageHeaderProps) {
  return (
    <div className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
      <div>
        {eyebrow ? (
          <Badge tone="cyan" className="mb-3">
            {eyebrow}
          </Badge>
        ) : null}
        <h1 className="text-2xl font-bold tracking-normal text-white md:text-3xl">{title}</h1>
        {description ? <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">{description}</p> : null}
      </div>
      {actions}
    </div>
  )
}
