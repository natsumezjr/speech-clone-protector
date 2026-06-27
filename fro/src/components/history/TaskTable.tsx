import { useState } from 'react'
import { Download, Eye, SlidersHorizontal, Trash2, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Badge } from '@/components/common/Badge'
import { Button } from '@/components/common/Button'
import { deleteTask, downloadProtectedAudio } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import type { HistoryTask, TaskStatus } from '@/types/task'
import { downloadBlob } from '@/utils/download'
import { seconds } from '@/utils/format'

type HistoryView = 'protection' | 'asr' | 'clone'

const statusTone: Record<TaskStatus, 'cyan' | 'green' | 'orange' | 'red'> = {
  queued: 'cyan',
  running: 'orange',
  completed: 'green',
  success: 'green',
  failed: 'red',
  error: 'red',
  cancelled: 'red',
}

const baseModeText: Record<string, string> = {
  standard: '标准',
  strong: '强',
  high_fidelity: '高保真',
  custom: '自定义',
  joint: '联合防护',
}

const targetText: Record<string, string> = {
  semantic: '语义保护',
  timbre: '特征防护',
  joint: '联合保护',
}

function formatDate(value?: string | null) {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  const pad = (input: number) => String(input).padStart(2, '0')
  return `${date.getFullYear()}.${date.getMonth() + 1}.${date.getDate()} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

function truncateFilename(filename: string, maxBase = 22) {
  if (!filename || filename === '-') return '-'
  const dot = filename.lastIndexOf('.')
  const base = dot > 0 ? filename.slice(0, dot) : filename
  const ext = dot > 0 ? filename.slice(dot) : ''
  if (base.length <= maxBase) return filename
  return `${base.slice(0, Math.max(8, maxBase - 8))}...${base.slice(-5)}${ext}`
}

function statusForView(task: HistoryTask, view: HistoryView): TaskStatus {
  const status = view === 'asr' ? task.asrStatus : view === 'clone' ? task.cloneStatus : task.protectionStatus
  if (status === 'queued' || status === 'running' || status === 'completed' || status === 'success' || status === 'failed' || status === 'error' || status === 'cancelled') return status
  return 'queued'
}

function progressForView(task: HistoryTask, view: HistoryView) {
  const status = statusForView(task, view)
  const progress = view === 'asr' ? task.asrProgress : view === 'clone' ? task.cloneProgress : task.protectionProgress
  const value = typeof progress === 'number' && Number.isFinite(progress) ? progress : status === 'completed' || status === 'success' ? 1 : 0
  return Math.max(0, Math.min(1, value))
}

function elapsedForView(task: HistoryTask, view: HistoryView) {
  const elapsed = view === 'asr' ? task.asrElapsedSec : view === 'clone' ? task.cloneElapsedSec : task.protectionElapsedSec
  return typeof elapsed === 'number' && Number.isFinite(elapsed) ? elapsed : null
}

function statusTextForView(task: HistoryTask, view: HistoryView) {
  const status = statusForView(task, view)
  const label = view === 'asr' ? 'ASR' : view === 'clone' ? '克隆' : '防护'
  if (status === 'queued') return `${label}排队中`
  if (status === 'running') return `${label}进行中`
  if (status === 'failed' || status === 'error') return `${label}失败`
  if (status === 'cancelled') return `${label}已取消`
  return `${label}完成`
}

function modelForView(task: HistoryTask, view: HistoryView) {
  if (view === 'asr') return task.asrModel
  if (view === 'clone') return task.cloneModel
  return task.processingModel
}

function modeLabel(task: HistoryTask) {
  const base = baseModeText[task.mode] ?? task.mode
  const target = task.targetMode ? targetText[task.targetMode] ?? task.targetMode : '联合保护'
  return `${base}${target}`
}

function ParamModal({ task, onClose }: { task: HistoryTask; onClose: () => void }) {
  const rows = [
    ['weightSemantic', task.parameters?.weightSemantic],
    ['weightIdentity', task.parameters?.weightIdentity ?? task.parameters?.weightFeature],
    ['weightPsy', task.parameters?.weightPsy],
    ['weightL2', task.parameters?.weightL2],
  ]
  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/70 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="自定义参数">
      <div className="ui-card w-full max-w-md p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-black text-white">自定义参数</h3>
            <p className="mt-1 text-xs text-slate-500">{truncateFilename(task.filename, 30)}</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="overflow-hidden rounded-[7px] border border-cyan-300/14">
          <table className="w-full text-sm">
            <tbody>
              {rows.map(([name, value]) => (
                <tr key={String(name)} className="border-b border-cyan-300/10 last:border-b-0">
                  <td className="px-4 py-3 font-mono text-xs text-slate-400">{name}</td>
                  <td className="px-4 py-3 text-right font-mono text-cyan-100">{typeof value === 'number' ? value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '') : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export function TaskTable({ tasks, view, onDeleted }: { tasks: HistoryTask[]; view: HistoryView; onDeleted?: () => void }) {
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)
  const [paramTask, setParamTask] = useState<HistoryTask | null>(null)

  const handleOpen = (task: HistoryTask) => {
    if (!task.protectedFilename || task.protectedFilename === '-') {
      pushToast({ kind: 'error', title: '暂无保护音频', description: '该历史记录还没有可打开的保护结果。' })
      return
    }
    navigate(`/results/${task.taskId}`)
  }

  const handleDownload = async (taskId: string) => {
    try {
      const { blob, filename } = await downloadProtectedAudio(taskId)
      downloadBlob(blob, filename)
      pushToast({ kind: 'success', title: '保护音频已开始下载' })
    } catch (error) {
      pushToast({ kind: 'error', title: '下载失败', description: error instanceof Error ? error.message : '请检查后端接口。' })
    }
  }

  const handleDelete = async (taskId: string) => {
    try {
      await deleteTask(taskId)
      pushToast({ kind: 'success', title: '历史记录已删除' })
      onDeleted?.()
    } catch (error) {
      pushToast({ kind: 'error', title: '删除失败', description: error instanceof Error ? error.message : '请检查后端接口。' })
    }
  }

  if (tasks.length === 0) {
    return <div className="ui-card p-10 text-center text-slate-400">暂无历史任务</div>
  }

  return (
    <div className="ui-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[1120px] text-center text-sm">
          <thead className="border-b border-cyan-300/10 bg-slate-950/32 text-xs text-slate-400">
            <tr>
              {['文件名', '模式', '状态', '进度', '模型名称', '起始时间', '处理时间', '操作'].map((head) => (
                <th key={head} className="px-4 py-3 font-medium">
                  {head}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tasks.map((task) => {
              const rowStatus = statusForView(task, view)
              const progress = progressForView(task, view)
              const elapsed = elapsedForView(task, view)
              const isCustom = task.mode === 'custom'
              return (
                <tr key={task.taskId} className="border-b border-cyan-300/8 hover:bg-cyan-400/[0.035]">
                  <td className="max-w-[220px] px-4 py-4 text-white" title={task.filename}>{truncateFilename(task.filename)}</td>
                  <td className="px-4 py-4 text-slate-300">
                    <div className="flex items-center justify-center gap-2">
                      <span>{modeLabel(task)}</span>
                      {isCustom ? (
                        <button type="button" onClick={() => setParamTask(task)} className="grid h-7 w-7 place-items-center rounded-[6px] border border-cyan-300/16 bg-white/[0.035] text-cyan-200" title="查看参数">
                          <SlidersHorizontal className="h-3.5 w-3.5" />
                        </button>
                      ) : null}
                    </div>
                  </td>
                  <td className="px-4 py-4">
                    <Badge tone={statusTone[rowStatus]}>{statusTextForView(task, view)}</Badge>
                  </td>
                  <td className="w-[160px] px-4 py-4">
                    <div className="mx-auto h-2 max-w-[140px] overflow-hidden rounded-full bg-slate-800">
                      <div className="h-full rounded-full bg-cyan-400 transition-all duration-300" style={{ width: `${Math.round(progress * 100)}%` }} />
                    </div>
                    <p className="mt-1 text-center font-mono text-xs text-slate-500">{Math.round(progress * 100)}%</p>
                  </td>
                  <td className="px-4 py-4 text-slate-300">{modelForView(task, view) || '-'}</td>
                  <td className="px-4 py-4 text-slate-400">{formatDate(task.createdAt)}</td>
                  <td className="px-4 py-4 text-slate-300">{elapsed !== null ? seconds(elapsed) : '-'}</td>
                  <td className="px-4 py-4">
                    <div className="flex justify-center gap-2">
                      <Button variant="ghost" className="h-9 px-2" title="查看结果" onClick={() => handleOpen(task)}>
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" className="h-9 px-2" title="下载保护音频" onClick={() => void handleDownload(task.taskId)}>
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button variant="danger" className="h-9 px-2" title="删除" onClick={() => void handleDelete(task.taskId)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {paramTask ? <ParamModal task={paramTask} onClose={() => setParamTask(null)} /> : null}
    </div>
  )
}
