import type { ReactNode } from 'react'
import { Panel } from '@/components/common/Panel'

interface MetricCardProps {
  label: string
  value: string
  description?: string
  icon?: ReactNode
  tone?: 'cyan' | 'green' | 'blue' | 'orange' | 'purple'
}

const toneText = {
  cyan: 'text-cyan-200',
  green: 'text-emerald-200',
  blue: 'text-blue-200',
  orange: 'text-amber-200',
  purple: 'text-violet-200',
}

export function MetricCard({ label, value, description, icon, tone = 'cyan' }: MetricCardProps) {
  return (
    <Panel className="p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="text-xs font-medium text-slate-400">{label}</span>
        {icon ? <span className={toneText[tone]}>{icon}</span> : null}
      </div>
      <div className={`text-2xl font-bold ${toneText[tone]}`}>{value}</div>
      {description ? <p className="mt-2 text-xs leading-5 text-slate-400">{description}</p> : null}
    </Panel>
  )
}
