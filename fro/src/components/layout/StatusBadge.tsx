import { ShieldCheck } from 'lucide-react'
import { Badge } from '@/components/common/Badge'

export function StatusBadge() {
  return (
    <Badge tone="green" className="hidden h-9 px-4 sm:inline-flex">
      <ShieldCheck className="h-3.5 w-3.5" />
      系统防护中
    </Badge>
  )
}
