import { CheckCircle2, Clock3, Download, FileArchive, FileText, Loader2 } from 'lucide-react'
import { Button } from '@/components/common/Button'
import { Panel } from '@/components/common/Panel'
import { downloadProtectedAudio } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { downloadBlob } from '@/utils/download'

const stages = ['文件预处理', '编码器加载', '扰动优化', '心理声学约束', '结果评估', '报告生成']

interface TaskProgressStepperProps {
  progress: number
  running: boolean
  taskId?: string
}

export function TaskProgressStepper({ progress, running, taskId = 'mock-task-001' }: TaskProgressStepperProps) {
  const pushToast = useAppStore((state) => state.pushToast)
  const activeIndex = Math.min(stages.length - 1, Math.floor(progress * stages.length))

  const download = async () => {
    try {
      const { blob, filename } = await downloadProtectedAudio(taskId)
      downloadBlob(blob, filename)
      pushToast({ kind: 'success', title: '保护音频已开始下载' })
    } catch (error) {
      pushToast({ kind: 'error', title: '下载失败', description: error instanceof Error ? error.message : '请稍后再试。' })
    }
  }

  return (
    <Panel className="mt-6">
      <div className="grid gap-6 lg:grid-cols-[1fr_330px]">
        <div>
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">任务状态</h2>
            <span className="text-sm text-cyan-100">{Math.round(progress * 100)}%</span>
          </div>
          <div className="grid gap-3 md:grid-cols-6">
            {stages.map((stage, index) => {
              const complete = progress >= (index + 1) / stages.length
              const active = running && index === activeIndex
              return (
                <div key={stage} className="rounded-xl border border-white/10 bg-slate-950/45 p-3">
                  <div className="mb-2">
                    {complete ? (
                      <CheckCircle2 className="h-5 w-5 text-emerald-300" />
                    ) : active ? (
                      <Loader2 className="h-5 w-5 animate-spin text-cyan-200" />
                    ) : (
                      <Clock3 className="h-5 w-5 text-slate-500" />
                    )}
                  </div>
                  <p className="text-sm font-medium text-white">{stage}</p>
                  <p className="mt-1 text-xs text-slate-400">{complete ? '已完成' : active ? '进行中' : '等待开始'}</p>
                </div>
              )
            })}
          </div>
        </div>
        <div>
          <h3 className="mb-4 text-sm font-semibold text-slate-200">结果产物</h3>
          <div className="space-y-3">
            <Button variant="secondary" className="w-full justify-start" icon={<Download className="h-4 w-4" />} onClick={download}>
              保护音频 .wav
            </Button>
            {[
              { label: '评估报告 .pdf', Icon: FileText },
              { label: '详细数据 .csv', Icon: FileText },
              { label: '证据包 .zip', Icon: FileArchive },
            ].map(({ label, Icon }) => (
              <Button
                key={label}
                variant="ghost"
                className="w-full justify-start border border-white/10 bg-white/5"
                icon={<Icon className="h-4 w-4" />}
                onClick={() => pushToast({ kind: 'info', title: '后端接口预留', description: `${label} 将在后端接入后导出。` })}
              >
                {label}
              </Button>
            ))}
          </div>
        </div>
      </div>
    </Panel>
  )
}
