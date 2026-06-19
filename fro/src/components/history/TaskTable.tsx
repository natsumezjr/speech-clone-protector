import { Download, Eye, Trash2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Badge } from '@/components/common/Badge'
import { Button } from '@/components/common/Button'
import { deleteTask, downloadProtectedAudio } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import type { HistoryTask, TaskStatus } from '@/types/task'
import { downloadBlob } from '@/utils/download'
import { percent } from '@/utils/format'

const statusText: Record<TaskStatus, string> = {
  queued: '排队中',
  running: '处理中',
  completed: '已完成',
  failed: '失败',
}

const statusTone: Record<TaskStatus, 'cyan' | 'green' | 'orange' | 'red'> = {
  queued: 'cyan',
  running: 'orange',
  completed: 'green',
  failed: 'red',
}

const modeText: Record<string, string> = {
  standard: '标准保护',
  strong: '强保护',
  high_fidelity: '高保真',
  custom: '高级自定义',
  joint: '联合防护',
}

export function TaskTable({ tasks }: { tasks: HistoryTask[] }) {
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)

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
      pushToast({ kind: 'success', title: '删除请求已提交', description: 'Mock 模式仅展示交互，Backend 模式会调用 DELETE 接口。' })
    } catch (error) {
      pushToast({ kind: 'error', title: '删除失败', description: error instanceof Error ? error.message : '请检查后端接口。' })
    }
  }

  if (tasks.length === 0) {
    return <div className="ui-card p-10 text-center text-slate-400">暂无匹配任务</div>
  }

  return (
    <div className="ui-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[980px] text-left text-sm">
          <thead className="border-b border-cyan-300/10 bg-slate-950/32 text-xs uppercase text-slate-400">
            <tr>
              {['任务 ID', '文件名', '防护模式', '数据模式', '状态', 'WER', 'Feature 相似度下降', 'PESQ', '创建时间', '操作'].map((head) => (
                <th key={head} className="px-4 py-3 font-medium">
                  {head}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tasks.map((task) => (
              <tr key={task.taskId} className="border-b border-cyan-300/8 hover:bg-cyan-400/[0.035]">
                <td className="px-4 py-4 font-mono text-xs text-cyan-100">{task.taskId}</td>
                <td className="px-4 py-4 text-white">{task.filename}</td>
                <td className="px-4 py-4 text-slate-300">{modeText[task.mode]}</td>
                <td className="px-4 py-4">
                  <Badge tone={task.dataMode === 'mock' ? 'cyan' : 'orange'}>{task.dataMode === 'mock' ? 'Mock' : 'Backend'}</Badge>
                </td>
                <td className="px-4 py-4">
                  <Badge tone={statusTone[task.status]}>{statusText[task.status]}</Badge>
                </td>
                <td className="px-4 py-4 text-slate-300">{task.wer ? percent(task.wer) : '-'}</td>
                <td className="px-4 py-4 text-slate-300">{task.simDropRate ? percent(task.simDropRate) : '-'}</td>
                <td className="px-4 py-4 text-slate-300">{task.pesq ? task.pesq.toFixed(2) : '-'}</td>
                <td className="px-4 py-4 text-slate-400">{task.createdAt}</td>
                <td className="px-4 py-4">
                  <div className="flex gap-2">
                    <Button variant="ghost" className="h-9 px-2" title="查看结果" onClick={() => navigate('/results/mock-task-001')}>
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
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
