import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { PageHeader } from '@/components/layout/PageHeader'
import { Panel } from '@/components/common/Panel'
import { Spinner } from '@/components/common/Spinner'
import { TaskFilters } from '@/components/history/TaskFilters'
import { TaskTable } from '@/components/history/TaskTable'
import { listTasks } from '@/services/apiClient'
import { useTaskStore } from '@/store/taskStore'

export function HistoryPage() {
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('all')
  const [mode, setMode] = useState('all')
  const setHistoryTasks = useTaskStore((state) => state.setHistoryTasks)
  const { data = [], isLoading, error } = useQuery({
    queryKey: ['tasks'],
    queryFn: async () => {
      const tasks = await listTasks()
      setHistoryTasks(tasks)
      return tasks
    },
  })

  const filtered = useMemo(
    () =>
      data.filter((task) => {
        const hitText = `${task.taskId} ${task.filename}`.toLowerCase().includes(search.toLowerCase())
        const hitStatus = status === 'all' || task.status === status
        const hitMode = mode === 'all' || task.mode === mode
        return hitText && hitStatus && hitMode
      }),
    [data, mode, search, status],
  )

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="Task History" title="历史任务" description="轻量化展示任务闭环，支持结果查看、保护音频下载与后端删除接口预留。" />
      <TaskFilters search={search} status={status} mode={mode} onSearchChange={setSearch} onStatusChange={setStatus} onModeChange={setMode} />
      {isLoading ? (
        <Panel className="grid min-h-60 place-items-center">
          <Spinner />
        </Panel>
      ) : error ? (
        <Panel className="border-red-400/25 bg-red-950/20 text-red-100">
          Backend 模式下获取历史任务失败：{error instanceof Error ? error.message : '未知错误'}
        </Panel>
      ) : (
        <TaskTable tasks={filtered} />
      )}
    </div>
  )
}
