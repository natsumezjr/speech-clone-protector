import { Badge } from '@/components/common/Badge'
import { Panel } from '@/components/common/Panel'
import type { TaskResult } from '@/types/task'

export function TaskSummaryStrip({ result }: { result: TaskResult }) {
  const items = [
    ['任务 ID', result.taskId],
    ['任务状态', '已完成'],
    ['完成时间', result.completedAt],
    ['处理耗时', `${result.elapsedSec}s`],
    ['防护模式', '联合防护（推荐）'],
    ['综合判定', result.verdict],
    ['综合得分', result.score.toFixed(1)],
  ]

  return (
    <Panel className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-7">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
          <p className="text-xs text-slate-400">{label}</p>
          {label === '综合判定' ? (
            <Badge tone="green" className="mt-2">
              {value}
            </Badge>
          ) : (
            <p className="mt-2 truncate text-sm font-semibold text-white">{value}</p>
          )}
        </div>
      ))}
    </Panel>
  )
}
