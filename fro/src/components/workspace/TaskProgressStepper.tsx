import { CheckCircle2, Clock3, Download, FileArchive, FileText, Loader2, ShieldCheck } from 'lucide-react'
import type { ReactNode } from 'react'
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
    <Panel className="border-sky-400/24 bg-[#071226]/88">
      <div className="grid gap-6 xl:grid-cols-[1fr_360px_190px]">
        <div>
          <div className="mb-5 flex items-center justify-between">
            <h2 className="text-2xl font-bold text-white">任务状态</h2>
            <span className="text-lg font-bold text-cyan-100">{Math.round(progress * 100)}%</span>
          </div>
          <div className="grid items-start gap-2 md:grid-cols-6">
            {stages.map((stage, index) => {
              const complete = progress >= (index + 1) / stages.length
              const active = running && index === activeIndex
              return (
                <div key={stage} className="relative">
                  {index < stages.length - 1 ? <div className="absolute left-[52%] top-6 hidden h-px w-[96%] bg-sky-400/38 md:block" /> : null}
                  <div className="relative flex flex-col items-center text-center">
                    <div className="grid h-12 w-12 place-items-center rounded-full border border-sky-400/30 bg-[#050a19]">
                      {complete ? (
                        <CheckCircle2 className="h-6 w-6 text-emerald-300" />
                      ) : active ? (
                        <Loader2 className="h-6 w-6 animate-spin text-cyan-200" />
                      ) : (
                        <Clock3 className="h-6 w-6 text-slate-500" />
                      )}
                    </div>
                    <p className="mt-3 text-sm font-semibold text-white">
                      {index + 1} {stage}
                    </p>
                    <p className="mt-1 text-xs text-slate-400">{complete ? '已完成' : active ? '进行中' : '等待开始'}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="rounded-xl border border-sky-400/24 bg-[#050a19]/70 p-4">
          <h3 className="mb-3 text-base font-bold text-white">结果产物（完成后可下载）</h3>
          <div className="grid grid-cols-2 gap-2">
            <ArtifactButton label="保护音频 .wav" icon={<Download className="h-4 w-4" />} onClick={download} primary />
            <ArtifactButton
              label="评估报告 .pdf"
              icon={<FileText className="h-4 w-4" />}
              onClick={() => pushToast({ kind: 'info', title: '后端接口预留', description: '评估报告 .pdf 将在后端接入后导出。' })}
            />
            <ArtifactButton
              label="详细数据 .csv"
              icon={<FileText className="h-4 w-4" />}
              onClick={() => pushToast({ kind: 'info', title: '后端接口预留', description: '详细数据 .csv 将在后端接入后导出。' })}
            />
            <ArtifactButton
              label="证据包 .zip"
              icon={<FileArchive className="h-4 w-4" />}
              onClick={() => pushToast({ kind: 'info', title: '后端接口预留', description: '证据包 .zip 将在后端接入后导出。' })}
            />
          </div>
        </div>

        <div className="grid place-items-center rounded-xl border border-white/10 bg-[#050a19]/70 p-4 text-center">
          <ShieldCheck className="mb-3 h-10 w-10 text-cyan-200" />
          <h3 className="font-bold text-white">操作日志</h3>
          <p className="mt-2 text-xs text-slate-400">查看任务详细日志</p>
        </div>
      </div>
    </Panel>
  )
}

function ArtifactButton({
  label,
  icon,
  onClick,
  primary,
}: {
  label: string
  icon: ReactNode
  onClick: () => void
  primary?: boolean
}) {
  return (
    <Button
      variant={primary ? 'secondary' : 'ghost'}
      className={`h-11 justify-start border border-white/10 ${primary ? 'bg-sky-400/16' : 'bg-white/[0.04]'}`}
      icon={icon}
      onClick={onClick}
    >
      {label}
    </Button>
  )
}
