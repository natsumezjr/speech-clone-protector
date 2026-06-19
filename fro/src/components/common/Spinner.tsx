import { cn } from '@/lib/utils'

export function Spinner({ className }: { className?: string }) {
  return <span className={cn('inline-block h-5 w-5 animate-spin rounded-full border-2 border-cyan-200 border-t-transparent', className)} />
}
