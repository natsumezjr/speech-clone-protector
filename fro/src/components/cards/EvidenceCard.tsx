import type { ReactNode } from 'react'
import { Panel } from '@/components/common/Panel'

interface EvidenceCardProps {
  title: string
  children: ReactNode
}

export function EvidenceCard({ title, children }: EvidenceCardProps) {
  return (
    <Panel className="border-sky-400/22 bg-[#071226]/88 p-6">
      <h2 className="mb-5 text-2xl font-bold text-white">{title}</h2>
      {children}
    </Panel>
  )
}
