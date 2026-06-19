import { Search } from 'lucide-react'

interface TaskFiltersProps {
  search: string
  status: string
  mode: string
  onSearchChange: (value: string) => void
  onStatusChange: (value: string) => void
  onModeChange: (value: string) => void
}

export function TaskFilters({ search, status, mode, onSearchChange, onStatusChange, onModeChange }: TaskFiltersProps) {
  return (
    <div className="grid gap-3 rounded-2xl border border-white/10 bg-slate-900/55 p-4 md:grid-cols-[1fr_180px_200px]">
      <label className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
        <input
          value={search}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder="搜索任务 ID 或文件名"
          className="w-full rounded-lg border border-white/10 bg-slate-950/70 py-2 pl-9 pr-3 text-sm text-white outline-none focus:border-cyan-300/50"
        />
      </label>
      <select
        value={status}
        onChange={(event) => onStatusChange(event.target.value)}
        className="rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-white"
      >
        <option value="all">全部状态</option>
        <option value="completed">已完成</option>
        <option value="running">处理中</option>
        <option value="failed">失败</option>
      </select>
      <select
        value={mode}
        onChange={(event) => onModeChange(event.target.value)}
        className="rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-white"
      >
        <option value="all">全部模式</option>
        <option value="standard">标准保护</option>
        <option value="strong">强保护</option>
        <option value="high_fidelity">高保真</option>
        <option value="joint">联合防护</option>
      </select>
    </div>
  )
}
