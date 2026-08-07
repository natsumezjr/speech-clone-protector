import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { Panel } from '@/components/common/Panel'
import { Spinner } from '@/components/common/Spinner'
import { TaskFilters } from '@/components/history/TaskFilters'
import { TaskTable } from '@/components/history/TaskTable'
import { listTasks } from '@/services/apiClient'
import { useTaskStore } from '@/store/taskStore'

type HistoryView = 'protection' | 'asr' | 'clone'

const historyViews: Array<{ key: HistoryView; label: string }> = [
  { key: 'protection', label: '保护历史记录' },
  { key: 'asr', label: 'ASR 历史记录' },
  { key: 'clone', label: '语音克隆历史记录' },
]

function taskStatusForView(task: { protectionStatus?: string | null; asrStatus?: string | null; cloneStatus?: string | null }, view: HistoryView) {
  if (view === 'asr') return task.asrStatus
  if (view === 'clone') return task.cloneStatus
  return task.protectionStatus
}

function matchesHistoryView(task: { protectionStatus?: string | null; asrStatus?: string | null; cloneStatus?: string | null; hasAsrResult?: boolean; hasCloneResult?: boolean }, view: HistoryView) {
  if (view === 'protection') return Boolean(task.protectionStatus)
  if (view === 'asr') return Boolean(task.asrStatus) || Boolean(task.hasAsrResult)
  return Boolean(task.cloneStatus) || Boolean(task.hasCloneResult)
}

export function HistoryPage() {
  const [searchParams] = useSearchParams()
  const initialView = searchParams.get('view')
  const [search, setSearch] = useState(searchParams.get('search') ?? '')
  const [status, setStatus] = useState('all')
  const [mode, setMode] = useState('all')
  const [view, setView] = useState<HistoryView>(initialView === 'asr' || initialView === 'clone' ? initialView : 'protection')
  const queryClient = useQueryClient()
  const setHistoryTasks = useTaskStore((state) => state.setHistoryTasks)
  const { data = [], isLoading, error } = useQuery({
    queryKey: ['tasks'],
    queryFn: async () => {
      const tasks = await listTasks()
      setHistoryTasks(tasks)
      return tasks
    },
    refetchInterval: (query) => {
      const tasks = query.state.data ?? []
      return tasks.some((task) => [task.protectionStatus, task.asrStatus, task.cloneStatus].some((item) => item === 'queued' || item === 'running')) ? 1000 : 5000
    },
  })

  const filtered = useMemo(
    () =>
      data.filter((task) => {
        const query = search.toLowerCase()
        const hitText = task.filename.toLowerCase().includes(query) || task.taskId.toLowerCase().includes(query)
        const hitStatus = status === 'all' || taskStatusForView(task, view) === status
        const hitMode = mode === 'all' || task.mode === mode
        return hitText && hitStatus && hitMode && matchesHistoryView(task, view)
      }),
    [data, mode, search, status, view],
  )

  return (
    <div className="space-y-5">
      <div className="ui-card flex gap-2 overflow-x-auto p-2">
        {historyViews.map((item) => {
          const count = data.filter((task) => matchesHistoryView(task, item.key)).length
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => setView(item.key)}
              className={`h-10 shrink-0 rounded-[7px] px-4 text-sm font-black transition ${view === item.key ? 'bg-cyan-400 text-slate-950' : 'border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white'}`}
            >
              {item.label}
              <span className="ml-2 font-mono text-xs opacity-70">{count}</span>
            </button>
          )
        })}
      </div>
      <TaskFilters search={search} status={status} mode={mode} onSearchChange={setSearch} onStatusChange={setStatus} onModeChange={setMode} />
      {isLoading ? (
        <Panel className="grid min-h-60 place-items-center">
          <Spinner />
        </Panel>
      ) : error ? (
        <Panel className="border-red-400/25 bg-red-950/20 text-red-100">获取历史任务失败：{error instanceof Error ? error.message : '未知错误'}</Panel>
      ) : (
        <TaskTable tasks={filtered} view={view} onChanged={() => void queryClient.invalidateQueries({ queryKey: ['tasks'] })} />
      )}
    </div>
  )
}
