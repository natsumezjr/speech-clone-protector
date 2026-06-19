import type { ReactNode } from 'react'
import { CheckCircle2 } from 'lucide-react'
import { Panel } from '@/components/common/Panel'

interface StrategyCardProps {
  title: string
  items: string[]
  icon: ReactNode
}

export function StrategyCard({ title, items, icon }: StrategyCardProps) {
  return (
    <Panel>
      <div className="mb-4 flex items-center gap-3">
        <div className="rounded-xl border border-cyan-300/20 bg-cyan-300/10 p-3 text-cyan-100">{icon}</div>
        <h3 className="text-lg font-semibold text-white">{title}</h3>
      </div>
      <div className="space-y-3">
        {items.map((item) => (
          <div key={item} className="flex gap-2 text-sm text-slate-300">
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-300" />
            <span>{item}</span>
          </div>
        ))}
      </div>
    </Panel>
  )
}
