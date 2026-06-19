import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AudioUploadPanel } from '@/components/audio/AudioUploadPanel'
import { PageHeader } from '@/components/layout/PageHeader'
import { ArchitectureOverview } from '@/components/workspace/ArchitectureOverview'
import { ParameterForm } from '@/components/workspace/ParameterForm'
import { TaskProgressStepper } from '@/components/workspace/TaskProgressStepper'
import { createProtectionTask, getTaskStatus } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { ProtectionTaskRequest } from '@/types/task'
import { isMockMode } from '@/config/runtime'

export function WorkspacePage() {
  const navigate = useNavigate()
  const timerRef = useRef<number | null>(null)
  const [progress, setProgress] = useState(0)
  const [running, setRunning] = useState(false)
  const [taskId, setTaskId] = useState<string>()
  const pushToast = useAppStore((state) => state.pushToast)
  const setCurrentTaskStatus = useTaskStore((state) => state.setCurrentTaskStatus)

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [])

  const startTask = async (payload: ProtectionTaskRequest) => {
    try {
      setRunning(true)
      setProgress(0)
      const created = await createProtectionTask(payload)
      setTaskId(created.taskId)
      pushToast({ kind: 'success', title: '任务已创建', description: isMockMode ? '正在推进 Mock 演示流程。' : `任务 ID：${created.taskId}` })

      timerRef.current = window.setInterval(async () => {
        setProgress((value) => {
          const next = Math.min(1, value + 0.08)
          if (next >= 1) {
            if (timerRef.current) window.clearInterval(timerRef.current)
            setRunning(false)
            void getTaskStatus(created.taskId).then(setCurrentTaskStatus)
            window.setTimeout(() => navigate(`/results/${created.taskId}`), 450)
          }
          return next
        })
      }, 420)
    } catch (error) {
      setRunning(false)
      pushToast({ kind: 'error', title: '任务创建失败', description: error instanceof Error ? error.message : '请检查后端服务。' })
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Protection Workspace"
        title="防护工作台"
        description="提交音频、配置语义防护与音色防护策略，并观察端到端防护任务如何生成证据链。"
      />
      <div className="grid gap-6 xl:grid-cols-[0.85fr_1.2fr_0.95fr]">
        <AudioUploadPanel />
        <ParameterForm onSubmitTask={(payload) => void startTask(payload)} running={running} />
        <ArchitectureOverview />
      </div>
      <TaskProgressStepper progress={progress} running={running} taskId={taskId} />
    </div>
  )
}
