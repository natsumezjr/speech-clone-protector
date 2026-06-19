import type { ReactNode } from 'react'
import { Panel } from '@/components/common/Panel'

interface EvidenceCardProps {
  title: string
  children: ReactNode
}

export function EvidenceCard({ title, children }: EvidenceCardProps) {
  return (
    <Panel>
      <h2 className="mb-4 text-lg font-semibold text-white">{title}</h2>
      {children}
    </Panel>
  )
}
