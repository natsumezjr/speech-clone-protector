import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  CheckCircle2,
  ClipboardList,
  Clock3,
  Copy,
  Download,
  FileArchive,
  FileText,
  Info,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TestTube2,
  Volume2,
  X,
} from 'lucide-react'
import { cloneVoice, downloadEvidenceZip, downloadProtectedAudio, exportReport, getCapabilities, getTaskResult, getTaskStatus, listTasks, runAsrEval } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { AsrEval, AsrMetrics, CapabilitiesResponse, CloneEval, CloneVoiceRequest, CloneVoiceResult, DiffOp, LossFinal, LossTrendPoint, ProtectionRuntimeConfig, RadarPoint, TaskResult, TaskStatusResponse } from '@/types/task'
import type { AudioFileMeta } from '@/types/audio'
import { downloadBlob } from '@/utils/download'
import { cn } from '@/lib/utils'
import { AudioPlayer } from '@/components/audio/AudioPlayer'
import { formatDurationSeconds, getAudioDuration, getAudioSource } from '@/utils/audio'
import { TrendChart } from '@/components/charts/TrendChart'
import { MathText } from '@/components/common/MathText'

const statusText: Record<TaskResult['status'], string> = {
  queued: '排队中',
  running: '处理中',
  completed: '已完成',
  success: '已完成',
  failed: '失败',
  error: '失败',
  cancelled: '已取消',
}

const modeText: Record<TaskResult['mode'], string> = {
  standard: '标准保护',
  strong: '强保护',
  high_fidelity: '高保真',
  custom: '自定义',
  joint: '联合防护',
}

function configFromCapabilities(capabilities: CapabilitiesResponse | undefined): ProtectionRuntimeConfig | undefined {
  if (!capabilities) return undefined
  if (capabilities.config) return capabilities.config
  if (capabilities.defaults && capabilities.ranges && capabilities.models) {
    return {
      defaults: capabilities.defaults,
      ranges: capabilities.ranges,
      models: capabilities.models,
      constraints: capabilities.constraints,
    }
  }
  return undefined
}

function optionValues(options?: ProtectionRuntimeConfig['models'][string]) {
  return (options ?? []).map((option) => (typeof option === 'string' ? option : option.value)).filter(Boolean)
}

type BackendSelectOption = {
  label: string
  value: string
  status?: string
  languages?: string[]
}

function backendOptionItems(options?: ProtectionRuntimeConfig['models'][string]) {
  return (options ?? [])
    .map((option) =>
      typeof option === 'string'
        ? { label: option, value: option }
        : {
            label: option.label ?? option.value,
            value: option.backendValue ?? option.value,
            status: option.status,
            languages: Array.isArray(option.languages) ? option.languages : undefined,
          },
    )
    .filter((option) => option.value)
}

const defaultCloneText =
  '今天的语音克隆测试包含自然停顿、连续短句和较长上下文。我们希望模型在保持语速稳定的同时复现说话人的音色、韵律和情绪变化，用来比较原始音频与保护音频在下游合成系统中的差异。'

const delay = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

export function ResultsPage() {
  const { taskId } = useParams()
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)
  const setCurrentTaskResult = useTaskStore((state) => state.setCurrentTaskResult)
  const missingTaskGuardRef = useRef(false)
  const [asrOverrideState, setAsrOverrideState] = useState<{ taskId: string; asr: AsrMetrics }>()
  const [taskInfoOpen, setTaskInfoOpen] = useState(false)
  const [downloadOpen, setDownloadOpen] = useState(false)
  const { data, isLoading, error } = useQuery({
    queryKey: ['task-result', taskId],
    queryFn: async () => {
      const result = await getTaskResult(taskId as string)
      setCurrentTaskResult(result)
      return result
    },
    enabled: Boolean(taskId),
    retry: false,
  })

  useEffect(() => {
    if (taskId || missingTaskGuardRef.current) return
    missingTaskGuardRef.current = true
    void listTasks()
      .then((tasks) => {
        const latest = tasks.find((task) => (task.status === 'completed' || task.status === 'success') && task.protectedFilename && task.protectedFilename !== '-')
        if (latest) {
          navigate(`/results/${latest.taskId}`, { replace: true })
          return
        }
        pushToast({
          id: 'results-missing-task-id',
          kind: 'error',
          title: '请先进行音频保护任务',
          description: '结果页需要有效的 taskId。',
          dedupeMs: 5000,
        })
        navigate('/workspace', { replace: true })
      })
      .catch(() => {
        pushToast({
          id: 'results-missing-task-id',
          kind: 'error',
          title: '请先进行音频保护任务',
          description: '结果页需要有效的 taskId。',
          dedupeMs: 5000,
        })
        navigate('/workspace', { replace: true })
      })
  }, [navigate, pushToast, taskId])

  if (!taskId) return null

  if (isLoading) {
    return (
      <div className="grid min-h-[520px] place-items-center">
        <div className="text-center">
          <Loader2 className="mx-auto h-9 w-9 animate-spin text-cyan-300" />
          <p className="mt-4 text-slate-300">正在加载结果证据链...</p>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return <div className="ui-card p-6 text-red-100">{error instanceof Error ? error.message : '无法获取任务结果。'}</div>
  }

  const asrOverride = asrOverrideState?.taskId === taskId ? asrOverrideState.asr : undefined
  const displayData = asrOverride ? { ...data, asr: asrOverride, asrEval: asrOverride, asrModel: asrOverride.model ?? data.asrModel } : data

  return (
    <div className="-mx-5 max-w-none space-y-5 pb-6">
      <SummaryBar result={displayData} onTaskInfoClick={() => setTaskInfoOpen(true)} onDownloadClick={() => setDownloadOpen(true)} />

      <AudioCompare result={displayData} onAsrUpdated={(asr) => setAsrOverrideState({ taskId, asr })} />
      {taskInfoOpen ? <TaskInfoModal result={displayData} onClose={() => setTaskInfoOpen(false)} /> : null}
      {downloadOpen ? <DownloadModal result={displayData} onClose={() => setDownloadOpen(false)} /> : null}
    </div>
  )
}

function SummaryBar({ result, onTaskInfoClick, onDownloadClick }: { result: TaskResult; onTaskInfoClick: () => void; onDownloadClick: () => void }) {
  return (
    <section className="ui-card grid min-h-[74px] grid-cols-[250px_180px_250px_170px_230px_minmax(260px,1fr)] items-center px-5 max-2xl:grid-cols-[1.05fr_0.78fr_1.08fr_0.75fr_1fr_1.42fr] max-xl:h-auto max-xl:grid-cols-3 max-xl:gap-y-4 max-xl:py-4">
      <SummaryItem icon={<ClipboardList />} label="任务 ID" value={result.taskId} copy buttonTitle="查看任务信息" onClick={onTaskInfoClick} />
      <SummaryItem icon={<ShieldCheck />} label="任务状态" value={statusText[result.status] ?? result.status} green={result.status === 'completed' || result.status === 'success'} />
      <SummaryItem icon={<Clock3 />} label="完成时间" value={result.completedAt ?? '-'} />
      <SummaryItem icon={<Clock3 />} label="处理耗时" value={typeof result.elapsedSec === 'number' ? formatElapsed(result.elapsedSec) : '-'} />
      <SummaryItem icon={<Sparkles />} label="防护模式" value={modeText[result.mode] ?? result.mode} green />
      <button type="button" onClick={onDownloadClick} className="flex h-full min-h-[58px] items-center justify-center gap-4 border-l border-cyan-300/10 pl-5 transition hover:bg-cyan-400/[0.035]">
        <ShieldCheck className="h-11 w-11 text-cyan-300" />
        <div className="min-w-0 text-left">
          <p className="text-[27px] font-black leading-none text-cyan-200">已生成防护报告</p>
          <p className="mt-1 text-xs text-slate-400">点击此处下载</p>
        </div>
      </button>
    </section>
  )
}

function formatElapsed(seconds: number) {
  const hh = Math.floor(seconds / 3600)
  const mm = Math.floor((seconds % 3600) / 60)
  const ss = seconds % 60
  return [hh, mm, ss].map((value) => String(value).padStart(2, '0')).join(':')
}

function SummaryItem({
  icon,
  label,
  value,
  green,
  copy,
  onClick,
  buttonTitle,
}: {
  icon: ReactNode
  label: string
  value: string
  green?: boolean
  copy?: boolean
  onClick?: () => void
  buttonTitle?: string
}) {
  const content = (
    <>
      <span className="text-slate-500 [&_svg]:h-4 [&_svg]:w-4">{icon}</span>
      <div className="min-w-0">
        <p className="text-xs text-slate-500">{label}</p>
        <p className={cn('mt-1 truncate text-[14px] font-bold text-slate-200', green && 'text-emerald-300')}>{value}</p>
      </div>
      {copy ? <Copy className="h-4 w-4 shrink-0 text-slate-500" /> : null}
    </>
  )
  const className = 'flex min-w-0 items-center justify-center gap-3 border-r border-cyan-300/10 px-4 whitespace-nowrap'
  return onClick ? (
    <button type="button" onClick={onClick} className={cn(className, 'h-full min-h-[58px] transition hover:bg-cyan-400/[0.035]')} title={buttonTitle}>
      {content}
    </button>
  ) : (
    <div className={className}>{content}</div>
  )
}

function SectionTitle({ children, info }: { children: ReactNode; info?: boolean }) {
  return (
    <h2 className="flex h-6 items-center gap-2 whitespace-nowrap text-[16px] font-black leading-none text-white">
      {children}
      {info ? <Info className="h-3.5 w-3.5 text-slate-500" /> : null}
    </h2>
  )
}

type ComparePanel = 'protect' | 'asr' | 'clone'
type EditLevel = 'word' | 'char'
type EditMetrics = {
  level: EditLevel
  werOrCer: number
  substituteRate: number
  insertRate: number
  deleteRate: number
  referenceLength: number
  substitutions: number
  insertions: number
  deletions: number
  totalErrors: number
  errorShares: {
    substituteShare: number
    insertShare: number
    deleteShare: number
  }
  diffOps: DiffOp[]
}

function AudioCompare({ result, onAsrUpdated }: { result: TaskResult; onAsrUpdated: (asr: AsrMetrics) => void }) {
  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const pushToast = useAppStore((state) => state.pushToast)
  const [activePanel, setActivePanel] = useState<ComparePanel>('protect')
  const [protectedObjectUrl, setProtectedObjectUrl] = useState<string>()
  const [cloneModalOpen, setCloneModalOpen] = useState(false)
  const [cloneLoading, setCloneLoading] = useState(false)
  const [cloneError, setCloneError] = useState<string>()
  const [cloneResult, setCloneResult] = useState<CloneVoiceResult | undefined>(() => result.cloneResults?.at(-1))
  const [cloneTaskStatus, setCloneTaskStatus] = useState<TaskStatusResponse | null>(null)
  const [asrModalOpen, setAsrModalOpen] = useState(false)
  const [asrLoading, setAsrLoading] = useState(false)
  const [asrError, setAsrError] = useState<string>()
  const [asrModel, setAsrModel] = useState(result.asrModel || result.asr.model || '')
  const [cloneForm, setCloneForm] = useState<CloneVoiceRequest>({
    text: result.asr.originalText || defaultCloneText,
    model: 'XTTS-v2',
    language: 'zh-cn',
    speed: 1,
    speakerPrompt: '',
  })
  const { data: capabilities } = useQuery({
    queryKey: ['capabilities'],
    queryFn: getCapabilities,
    staleTime: 60_000,
  })
  const runtimeConfig = configFromCapabilities(capabilities)
  const configuredAsrOptions = useMemo(() => optionValues(runtimeConfig?.models.asr), [runtimeConfig?.models.asr])
  const asrOptions = useMemo(
    () => (configuredAsrOptions.length ? configuredAsrOptions : [result.asrModel || result.asr.model || 'openai/whisper-small']),
    [configuredAsrOptions, result.asr.model, result.asrModel],
  )
  const configuredTtsOptions = useMemo(
    () => backendOptionItems(runtimeConfig?.models.tts).filter((option) => option.status === undefined || option.status === 'available'),
    [runtimeConfig?.models.tts],
  )
  const ttsModelOptions = useMemo(
    () => (configuredTtsOptions.length ? configuredTtsOptions : [{ label: 'XTTS-v2', value: 'tts_models/multilingual/multi-dataset/xtts_v2' }]),
    [configuredTtsOptions],
  )
  const ttsOptions = useMemo(() => ttsModelOptions.map((option) => option.value), [ttsModelOptions])
  const selectedTtsOption = ttsModelOptions.find((option) => option.value === cloneForm.model) ?? ttsModelOptions[0]
  const cloneLanguages = useMemo(
    () => (selectedTtsOption?.languages?.length ? selectedTtsOption.languages : runtimeConfig?.clone?.languages?.length ? runtimeConfig.clone.languages : ['zh-cn', 'en']),
    [runtimeConfig, selectedTtsOption],
  )
  const cloneSpeeds = useMemo(
    () => (runtimeConfig?.clone?.speeds?.length ? runtimeConfig.clone.speeds : [0.75, 1, 1.25]),
    [runtimeConfig],
  )
  const defaultCloneConfig = runtimeConfig?.clone?.defaults
  const originalAudio = {
    ...result.originalAudio,
    objectUrl: result.originalAudio.objectUrl ?? uploadedFile?.objectUrl,
    audioUrl: result.originalAudio.audioUrl ?? uploadedFile?.audioUrl,
  }
  const protectedAudio = { ...result.protectedAudio, objectUrl: result.protectedAudio.objectUrl ?? protectedObjectUrl }
  const activeAsrEval = result.asrEval
  const originalText = activeAsrEval?.originalText ?? ''
  const referenceText = activeAsrEval?.referenceText ?? originalText
  const protectedText = activeAsrEval?.protectedText ?? ''
  const asrLevel = activeAsrEval?.metricLevel === 'word' || activeAsrEval?.metricLevel === 'char' ? activeAsrEval.metricLevel : chooseEditLevel(referenceText, protectedText)
  const asrEditStats = activeAsrEval && referenceText && protectedText ? computeEditMetrics(referenceText, protectedText, asrLevel) : null
  const activeCloneEval = cloneResult?.cloneEval ?? cloneResultToEval(cloneResult) ?? result.cloneEval ?? null
  const cloneModel = activeCloneEval?.cloneModel ?? '未生成'
  const speakerEvalModel = formatSpeakerEvalModel(
    activeCloneEval?.speakerEvalModel
      ?? result.metricSources?.['cloneEval.*']?.source
      ?? result.metricSources?.['speaker.*']?.source
      ?? activeCloneEval?.speakerModel
      ?? 'ECAPA-TDNN',
  )
  const compareTabs = [
    {
      key: 'protect',
      label: '保护',
      modelTitle: result.processingModel ?? result.generation?.source ?? '未生成',
    },
    {
      key: 'asr',
      label: 'ASR',
      modelTitle: activeAsrEval?.model ?? activeAsrEval?.asrModel ?? result.asrModel ?? '未生成',
    },
    {
      key: 'clone',
      label: '克隆',
      modelTitle: `克隆 ${cloneModel} · 评估 ${speakerEvalModel}`,
    },
  ] as const
  const activeModelTitle = compareTabs.find((tab) => tab.key === activePanel)?.modelTitle ?? '未生成'

  useEffect(() => {
    return () => {
      if (protectedObjectUrl?.startsWith('blob:')) URL.revokeObjectURL(protectedObjectUrl)
    }
  }, [protectedObjectUrl])

  useEffect(() => {
    const originalUrl = result.originalAudio.objectUrl
    const protectedUrl = result.protectedAudio.objectUrl
    const originalCloneUrl = cloneResult?.originalCloneAudio.objectUrl
    const protectedCloneUrl = cloneResult?.protectedCloneAudio.objectUrl
    return () => {
      if (originalUrl?.startsWith('blob:')) URL.revokeObjectURL(originalUrl)
      if (protectedUrl?.startsWith('blob:')) URL.revokeObjectURL(protectedUrl)
      if (originalCloneUrl?.startsWith('blob:')) URL.revokeObjectURL(originalCloneUrl)
      if (protectedCloneUrl?.startsWith('blob:')) URL.revokeObjectURL(protectedCloneUrl)
    }
  }, [cloneResult?.originalCloneAudio.objectUrl, cloneResult?.protectedCloneAudio.objectUrl, result.originalAudio.objectUrl, result.protectedAudio.objectUrl])

  useEffect(() => {
    if (!runtimeConfig) return
    const timeoutId = window.setTimeout(() => {
      setAsrModel((current) => (asrOptions.includes(current) ? current : runtimeConfig.defaults.semantic.asrModel || asrOptions[0]))
      const nextModel = defaultCloneConfig?.backendValue || defaultCloneConfig?.model || ttsOptions[0] || 'tts_models/multilingual/multi-dataset/xtts_v2'
      const nextModelOption = ttsModelOptions.find((option) => option.value === nextModel) ?? ttsModelOptions[0]
      const preferredLanguage = defaultCloneConfig?.uiPreferredLanguage || defaultCloneConfig?.language || 'zh-cn'
      const nextModelLanguages = nextModelOption?.languages?.length ? nextModelOption.languages : cloneLanguages
      const nextLanguage = nextModelLanguages.includes(preferredLanguage) ? preferredLanguage : nextModelLanguages[0] || 'en'
      const nextSpeed = defaultCloneConfig?.speed ?? cloneSpeeds[0] ?? 1
      setCloneForm((current) => {
        const currentModelOption = ttsModelOptions.find((option) => option.value === current.model)
        const currentLanguages = currentModelOption?.languages?.length ? currentModelOption.languages : cloneLanguages
        const currentLanguageSupported = currentLanguages.includes(current.language ?? '')
        const model = currentModelOption && currentLanguageSupported ? current.model : nextModel
        const modelOption = ttsModelOptions.find((option) => option.value === model) ?? nextModelOption
        const modelLanguages = modelOption?.languages?.length ? modelOption.languages : cloneLanguages
        return {
          ...current,
          model,
          language: modelLanguages.includes(current.language ?? '') ? current.language : nextLanguage,
          speed: cloneSpeeds.includes(Number(current.speed)) ? current.speed : nextSpeed,
        }
      })
    }, 0)
    return () => window.clearTimeout(timeoutId)
  }, [asrOptions, cloneLanguages, cloneSpeeds, defaultCloneConfig, runtimeConfig, ttsModelOptions, ttsOptions])

  const submitAsrTest = async () => {
    if (!asrOptions.includes(asrModel)) {
      setAsrError('请选择后端支持的 ASR 模型。')
      setAsrModalOpen(true)
      return
    }
    try {
      setAsrLoading(true)
      setAsrError(undefined)
      setAsrModalOpen(false)
      setActivePanel('asr')
      const response = await runAsrEval(result.taskId, { model: asrModel, referenceText: referenceText || originalText || result.asr.referenceText || result.asr.originalText || undefined })
      const asr = response.asr ?? (await waitForAsrEvalResult(result.taskId))
      onAsrUpdated(asr)
      pushToast({ kind: 'success', title: 'ASR 测试完成', description: asr.model ?? asrModel })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'ASR 测试失败，请检查后端服务。'
      setAsrError(message)
      setAsrModalOpen(true)
      pushToast({ kind: 'error', title: 'ASR 测试失败', description: message })
    } finally {
      setAsrLoading(false)
    }
  }

  const waitForAsrEvalResult = async (taskId: string) => {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const status = await getTaskStatus(taskId)
      if (status.asrResult?.asr) {
        const asrStatus = status.asrResult.asr.status
        if (asrStatus === 'unavailable' || asrStatus === 'failed' || asrStatus === 'error') {
          throw new Error(status.asrResult.asr.error || 'ASR 测试失败，请检查后端模型或依赖。')
        }
        return status.asrResult.asr
      }
      if ((status.status === 'failed' || status.status === 'error') && status.stage === 'asr_eval') {
        throw new Error(typeof status.error === 'string' ? status.error : status.message || 'ASR 测试失败，请检查后端服务。')
      }
      if ((status.status === 'completed' || status.status === 'success') && status.stage === 'asr_eval') {
        const latest = await getTaskResult(taskId)
        return latest.asr
      }
      await delay(1000)
    }
    throw new Error('ASR 测试仍在执行，请稍后刷新结果页查看。')
  }

  const loadProtectedAudio = async () => {
    if (getAudioSource(protectedAudio)) return getAudioSource(protectedAudio)
    const { blob } = await downloadProtectedAudio(result.taskId)
    const objectUrl = URL.createObjectURL(blob)
    setProtectedObjectUrl((current) => {
      if (current?.startsWith('blob:')) URL.revokeObjectURL(current)
      return objectUrl
    })
    return objectUrl
  }

  const validateCloneForm = () => {
    if (!cloneForm.text.trim()) return '请输入用于语音克隆的文本。'
    if (!ttsOptions.includes(cloneForm.model)) return '请选择后端支持的克隆模型。'
    if (!cloneLanguages.includes(cloneForm.language ?? '')) return '请选择后端支持的克隆语言。'
    if (!cloneSpeeds.includes(Number(cloneForm.speed))) return '请选择后端支持的克隆语速。'
    return undefined
  }

  const submitCloneTest = async () => {
    const validationError = validateCloneForm()
    if (validationError) {
      setCloneError(validationError)
      setCloneModalOpen(true)
      return
    }

    try {
      setCloneLoading(true)
      setCloneError(undefined)
      setCloneTaskStatus(null)
      setCloneModalOpen(false)
      setActivePanel('clone')
      const response = await cloneVoice(result.taskId, { ...cloneForm, text: cloneForm.text.trim() })
      const nextResult =
        (response.status === 'completed' || response.status === 'success') && getAudioSource(response.originalCloneAudio) && getAudioSource(response.protectedCloneAudio)
          ? response
          : await waitForCloneResult(result.taskId)
      setCloneResult(nextResult)
      pushToast({ kind: 'success', title: '语音克隆测试完成', description: nextResult.message ?? nextResult.cloneId })
    } catch (error) {
      const message = error instanceof Error ? error.message : '语音克隆测试失败，请检查表单或后端服务。'
      setCloneError(message)
      setCloneModalOpen(true)
      pushToast({ kind: 'error', title: '语音克隆测试失败', description: message })
    } finally {
      setCloneLoading(false)
    }
  }

  const waitForCloneResult = async (taskId: string) => {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const status = await getTaskStatus(taskId)
      if (status.stage === 'downstream_tts_eval') setCloneTaskStatus(status)
      if (status.cloneResult) {
        const latest = await getTaskResult(taskId)
        const latestClone = latest.cloneResults?.at(-1)
        if (latestClone) return latestClone
        return status.cloneResult
      }
      if ((status.status === 'failed' || status.status === 'error') && status.stage === 'downstream_tts_eval') {
        throw new Error(typeof status.error === 'string' ? status.error : status.message || '语音克隆测试失败，请检查后端服务。')
      }
      if ((status.status === 'completed' || status.status === 'success') && status.stage === 'downstream_tts_eval') {
        const latest = await getTaskResult(taskId)
        const latestClone = latest.cloneResults?.at(-1)
        if (latestClone) return latestClone
      }
      await delay(1000)
    }
    throw new Error('语音克隆测试仍在执行，请稍后刷新结果页查看。')
  }

  return (
    <section className="ui-card h-full p-5">
      <div className="grid grid-cols-[auto_minmax(180px,1fr)_auto] items-center gap-3 max-xl:grid-cols-1 max-xl:items-start">
        <div className="flex flex-wrap items-center gap-3">
          <SectionTitle info>结果对比</SectionTitle>
          <div className="flex items-center gap-2">
            {compareTabs.map(({ key, label }) => (
              <button key={key} type="button" onClick={() => setActivePanel(key as ComparePanel)} className={cn('h-9 rounded-[7px] border border-cyan-300/14 px-3 text-sm font-black text-slate-300 transition hover:text-white', activePanel === key && 'bg-cyan-400/14 text-cyan-200')} title={`查看${label}结果`}>
                {label}
              </button>
            ))}
          </div>
        </div>
        <p className="min-w-0 truncate text-center text-xs font-black tracking-normal text-cyan-100/85 max-xl:text-left" title={activeModelTitle}>
          {activeModelTitle}
        </p>
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => setCloneModalOpen(true)} className="inline-flex h-9 items-center gap-2 rounded-[7px] border border-cyan-300/18 bg-cyan-400/10 px-3 text-sm font-black text-cyan-100 hover:bg-cyan-400/16">
            <TestTube2 className="h-4 w-4" />
            语音克隆测试
          </button>
          <button type="button" onClick={() => setAsrModalOpen(true)} className="inline-flex h-9 items-center gap-2 rounded-[7px] border border-cyan-300/18 bg-cyan-400/10 px-3 text-sm font-black text-cyan-100 hover:bg-cyan-400/16">
            {asrLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <TestTube2 className="h-4 w-4" />}
            ASR 测试
          </button>
        </div>
      </div>
      <div className="mt-5">
        {activePanel === 'protect' ? (
          <ProtectTab result={result} originalAudio={originalAudio} protectedAudio={protectedAudio} onProtectedPlayRequest={loadProtectedAudio} />
        ) : null}
        {activePanel === 'asr' ? (
          <AsrTab result={result} asrEval={activeAsrEval} editStats={asrEditStats} />
        ) : null}
        {activePanel === 'clone' ? (
          <CloneTab result={result} cloneEval={activeCloneEval} loading={cloneLoading} status={cloneTaskStatus} />
        ) : null}
      </div>
      {cloneModalOpen ? (
        <CloneVoiceModal
          form={cloneForm}
          error={cloneError}
          loading={cloneLoading}
          modelOptions={ttsModelOptions}
          languageOptions={cloneLanguages}
          speedOptions={cloneSpeeds}
          onChange={setCloneForm}
          onClose={() => setCloneModalOpen(false)}
          onSubmit={() => void submitCloneTest()}
        />
      ) : null}
      {asrModalOpen ? (
        <AsrEvalModal
          model={asrModel}
          error={asrError}
          loading={asrLoading}
          modelOptions={asrOptions}
          onChange={setAsrModel}
          onClose={() => setAsrModalOpen(false)}
          onSubmit={() => void submitAsrTest()}
        />
      ) : null}
    </section>
  )
}

function ProtectTab({
  result,
  originalAudio,
  protectedAudio,
  onProtectedPlayRequest,
}: {
  result: TaskResult
  originalAudio: AudioFileMeta
  protectedAudio: AudioFileMeta
  onProtectedPlayRequest: () => Promise<string | undefined>
}) {
  const perturbation = result.perturbation
  const quality = result.protectionQuality ?? result.quality
  const snr = optionalNumber(perturbation?.snr) ?? optionalNumber(quality?.snr)
  const epsilonUsageRate = optionalNumber(perturbation?.epsilonUsageRate) ?? computeEpsilonUsageRate(perturbation)

  return (
    <div className="space-y-5">
      <div className="grid items-center gap-6 pr-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
        <AudioCard title="原始音频（未保护）" audio={originalAudio} color="#00aef0" />
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-cyan-300/28 bg-slate-950/70 text-[18px] font-black text-white shadow-[0_0_24px_rgba(56,189,248,0.12)]">VS</div>
        <AudioCard title="保护音频（已防护）" audio={protectedAudio} color="#22c55e" green onPlayRequest={onProtectedPlayRequest} />
      </div>
      <div className="grid grid-cols-[minmax(360px,0.86fr)_minmax(520px,1.14fr)] items-stretch gap-5 max-xl:grid-cols-1">
        <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
          <SectionTitle>扰动与可听性分析</SectionTitle>
          <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-3">
            <ScoreBox label="扰动强度（L2 / 能量）" value={formatMetricValue(perturbation?.l2Norm ?? result.quality.l2Norm, 'loss')} />
            <ScoreBox label="扰动上限利用率" value={formatMetricValue(epsilonUsageRate, 'percent')} />
            <ScoreBox label="频谱差异 / SNR" value={formatMetricValue(snr, 'db')} />
          </div>
          <QualityPanel result={result} embedded />
        </section>
        <PsychoacousticPanel result={result} />
      </div>
      <div className="grid min-h-[380px] grid-cols-[minmax(0,1fr)_minmax(360px,0.72fr)] items-stretch gap-5 max-xl:grid-cols-1">
        <TrendPanel result={result} embedded />
        <InsightPanel title="保护结果解读" items={generateProtectionInsights(result)} />
      </div>
    </div>
  )
}

function AsrTab({ result, asrEval, editStats }: { result: TaskResult; asrEval?: AsrEval | null; editStats: EditMetrics | null }) {
  if (!asrEval) {
    return (
      <EmptyState
        title="未执行 ASR 测试"
        text="ASR 评估属于可选下游测试，点击右上角“ASR 测试”后显示转写差异与语义链路指标。"
      />
    )
  }

  const originalText = asrEval.originalText ?? ''
  const referenceText = asrEval.referenceText ?? originalText
  const protectedText = asrEval.protectedText ?? ''
  const diffOps = asrEval.diffOps ?? editStats?.diffOps ?? []
  const substituteRate = asrEval.substituteRate ?? editStats?.substituteRate
  const insertRate = asrEval.insertRate ?? editStats?.insertRate
  const wer = asrEval.wer ?? (editStats?.level === 'word' ? editStats.werOrCer : undefined)
  const cer = asrEval.cer ?? (editStats?.level === 'char' ? editStats.werOrCer : undefined)
  const tokenDiff = asrEval.tokenChangeRate ?? asrEval.tokenErrorRate
  const tokenUsesEditDistance = asrEval.tokenChangeRate == null && asrEval.tokenErrorRate != null
  const tokenReason = tokenUsesEditDistance ? '使用 token edit distance；可能受 token 序列长度差异影响。' : metricReason(result, ['asrEval.tokenChangeRate', 'asrEval.tokenErrorRate'])
  const semanticSourceInfo = metricSource(result, ['asrEval.semanticDrift'])
  const semanticIsMfccProxy = String(semanticSourceInfo?.source ?? '').toLowerCase() === 'mfcc_proxy'
  const semanticFoot = semanticIsMfccProxy ? 'MFCC proxy，仅代表声学特征漂移，不等同于 S3/HuBERT/Whisper 语义 encoder 漂移。' : undefined
  const semanticCardLabel = semanticIsMfccProxy ? 'SD（MFCC 代理）' : 'SD（语义漂移）'
  const semanticDetailLabel = semanticIsMfccProxy ? 'MFCC 代理漂移' : '语义表示漂移'
  const errorShares = asrErrorShares(asrEval, editStats)

  return (
    <div className="space-y-5">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_288px_minmax(0,1fr)]">
        <TextBox title="参考文本 / 原始转写（ASR）" text={referenceText || '未生成'} foot="用于 WER/CER 与 diff 的参考文本" />
        <div className="grid grid-cols-2 content-center gap-3">
          <ScoreBox label="WER（词错率）" value={formatMetricValue(wer, 'percent')} red compact />
          <ScoreBox label="CER（字错率）" value={formatMetricValue(cer, 'percent')} red compact />
          <ScoreBox label="Token 变化率" value={formatMetricValue(tokenDiff, 'percent')} foot={tokenDiff == null || tokenUsesEditDistance ? tokenReason : undefined} red compact />
          <ScoreBox label={semanticCardLabel} value={formatMetricValue(asrEval.semanticDrift, 'number')} foot={semanticFoot} red compact />
          <ScoreBox label="IR（插入率）" value={formatMetricValue(insertRate, 'percent')} red compact />
          <ScoreBox label="SR（替换率）" value={formatMetricValue(substituteRate, 'percent')} red compact />
        </div>
        <TextBox title="保护音频转写（ASR）" text={protectedText || '未生成'} foot="红色为新增内容，绿色删除线为原文缺失内容" content={diffOps.length ? renderDiffOps(diffOps) : undefined} />
      </div>
      <div className="grid grid-cols-[1.05fr_0.95fr] gap-5 max-lg:grid-cols-1">
        <MetricPanel title="语义链路分析">
          <ScoreBox label={semanticDetailLabel} value={formatMetricValue(asrEval.semanticDrift, 'number')} foot={semanticFoot} />
          <ScoreBox label="Token 变化率" value={formatMetricValue(tokenDiff, 'percent')} foot={tokenDiff == null || tokenUsesEditDistance ? tokenReason : undefined} />
          <ScoreBox label="指标层级" value={asrEval.metricLevel ?? editStats?.level ?? '未生成'} />
        </MetricPanel>
        <RateBreakdown substituteShare={errorShares?.substituteShare} insertShare={errorShares?.insertShare} />
      </div>
      <InsightPanel title="ASR 结果解读" items={generateAsrInsights(asrEval, editStats, result)} />
    </div>
  )
}

function CloneTab({ result, cloneEval, loading, status }: { result: TaskResult; cloneEval?: CloneEval | null; loading: boolean; status: TaskStatusResponse | null }) {
  if (loading) {
    return (
      <div className="grid items-center gap-6 pl-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
        <LoadingCard title="克隆原语音" progress={status?.stage === 'downstream_tts_eval' ? status.progress : undefined} message={status?.stage === 'downstream_tts_eval' ? status.message : undefined} />
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
        <LoadingCard title="克隆保护语音" progress={status?.stage === 'downstream_tts_eval' ? status.progress : undefined} message={status?.stage === 'downstream_tts_eval' ? status.message : undefined} />
      </div>
    )
  }

  if (!cloneEval) {
    return (
      <EmptyState
        title="未执行语音克隆测试"
        text="语音克隆评估属于可选下游测试，可能耗时较长。点击右上角“语音克隆测试”后显示克隆音频与声纹相似度结果。"
      />
    )
  }

  const cloneReason = cloneEval.reason ? shortMetricReason(cloneEval.reason) : metricReason(result, ['cloneEval.*'])

  return (
    <div className="space-y-5">
      {cloneReason ? <MetricNotice text={`克隆指标未生成原因：${cloneReason}`} /> : null}
      <div className="grid items-center gap-6 pl-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
        {cloneEval.originalCloneAudio ? <AudioCard title="克隆原语音" audio={cloneEval.originalCloneAudio} color="#a78bfa" /> : <EmptyMetricCard title="克隆原语音" text="后端未返回克隆原语音" />}
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
        {cloneEval.protectedCloneAudio ? <AudioCard title="克隆保护语音" audio={cloneEval.protectedCloneAudio} color="#f59e0b" /> : <EmptyMetricCard title="克隆保护语音" text="后端未返回克隆保护语音" />}
      </div>
      <div className="grid items-stretch grid-cols-[minmax(420px,0.95fr)_minmax(520px,1.05fr)] gap-5 max-xl:grid-cols-1">
        <div className="flex h-full flex-col gap-5">
          <CloneIdentityPanel cloneEval={cloneEval} />
          <CloneResultPanel cloneEval={cloneEval} />
        </div>
        <CloneVisualizationPanel result={result} cloneEval={cloneEval} />
      </div>
      <div className="grid grid-cols-1 gap-5">
        <InsightPanel title="克隆结果解读" items={generateCloneInsights(cloneEval)} />
      </div>
    </div>
  )
}

function LoadingCard({ title, progress, message }: { title: string; progress?: number; message?: string }) {
  return (
    <div className="grid h-[252px] place-items-center rounded-[9px] border border-violet-300/18 bg-violet-400/8 p-5 text-center">
      <div className="w-full">
        <Loader2 className="mx-auto h-9 w-9 animate-spin text-violet-200" />
        <p className="mt-4 text-sm font-black text-slate-100">{title}</p>
        {progress !== undefined ? (
          <div className="mt-3 mx-auto max-w-[180px]">
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-violet-400 transition-all duration-500" style={{ width: `${Math.max(4, Math.round(progress * 100))}%` }} />
            </div>
            <p className="mt-1 font-mono text-[10px] text-slate-400">{Math.round(progress * 100)}%</p>
          </div>
        ) : null}
        {message ? <p className="mt-2 text-xs text-slate-400">{message}</p> : <p className="mt-2 text-xs text-slate-400">正在等待后端返回克隆音频...</p>}
      </div>
    </div>
  )
}

function EmptyMetricCard({ title, text }: { title: string; text: string }) {
  return (
    <div className="grid h-[252px] place-items-center rounded-[9px] border border-violet-300/18 bg-slate-950/18 p-5 text-center">
      <div>
        <TestTube2 className="mx-auto h-9 w-9 text-violet-200" />
        <p className="mt-4 text-sm font-black text-slate-100">{title}</p>
        <p className="mt-2 text-xs text-slate-400">{text}</p>
      </div>
    </div>
  )
}

function EmptyState({ title, text }: { title: string; text: string }) {
  return (
    <div className="rounded-[9px] border border-cyan-300/12 bg-slate-950/18 p-6 text-center">
      <TestTube2 className="mx-auto h-9 w-9 text-cyan-200" />
      <h3 className="mt-4 text-base font-black text-slate-100">{title}</h3>
      <p className="mx-auto mt-2 max-w-[680px] text-sm leading-6 text-slate-400">{text}</p>
    </div>
  )
}

function MetricNotice({ text }: { text: string }) {
  return (
    <div className="rounded-[7px] border border-amber-300/20 bg-amber-300/[0.06] px-3 py-2 text-xs leading-5 text-amber-100">
      {text}
    </div>
  )
}

function MetricPanel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <SectionTitle>{title}</SectionTitle>
      <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3">{children}</div>
    </section>
  )
}

function RateBreakdown({ substituteShare, insertShare }: { substituteShare?: number | null; insertShare?: number | null }) {
  const rows = [
    ['替换占比', substituteShare, 'bg-yellow-300'],
    ['插入占比', insertShare, 'bg-red-300'],
  ] as const
  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <SectionTitle>错误类型占比</SectionTitle>
      <div className="mt-5 space-y-4">
        {rows.map(([label, value, color]) => {
          const numberValue = optionalNumber(value)
          return (
            <div key={label}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-bold text-slate-300">{label}</span>
                <span className="font-mono text-slate-400">{formatMetricValue(numberValue, 'percent')}</span>
              </div>
              {numberValue === null ? (
                <div className="rounded-[6px] border border-dashed border-cyan-300/12 px-3 py-2 text-xs text-slate-500">未生成</div>
              ) : (
                <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                  <div className={cn('h-full rounded-full', color)} style={{ width: `${Math.max(2, Math.min(100, numberValue * 100))}%` }} />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}

function CloneIdentityPanel({ cloneEval }: { cloneEval: CloneEval }) {
  const similarityDropRate = optionalNumber(cloneEval.similarityDropRate) ?? computeDropRate(cloneEval.originalSimilarity, cloneEval.protectedSimilarity)
  const embeddingIncreaseRate = optionalNumber(cloneEval.embeddingDistanceIncreaseRate) ?? computeRateChange(cloneEval.embeddingDistanceBefore, cloneEval.embeddingDistanceAfter)
  const confidenceBefore = optionalNumber(cloneEval.cloneConfidenceBefore)
  const confidenceAfter = optionalNumber(cloneEval.cloneConfidenceAfter)
  const confidenceDropRate = optionalNumber(cloneEval.cloneConfidenceDropRate)
  const hasCloneConfidence = confidenceBefore !== null || confidenceAfter !== null || confidenceDropRate !== null

  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <SectionTitle>声音身份特征链路分析</SectionTitle>
      <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(210px,1fr))] gap-3">
        <DeltaStatCard
          title="Speaker Similarity（越低越好）"
          before={formatMetricValue(cloneEval.originalSimilarity, 'number')}
          after={formatMetricValue(cloneEval.protectedSimilarity, 'number')}
          delta={`↓ ${formatRatioPercent(similarityDropRate, { clampToUnit: true })}`}
          foot="来源：后端声纹相似度评估"
          tone="green"
        />
        <DeltaStatCard
          title="Embedding 距离（cosine distance，越大越好）"
          before={formatMetricValue(cloneEval.embeddingDistanceBefore, 'number')}
          after={formatMetricValue(cloneEval.embeddingDistanceAfter, 'number')}
          delta={`↑ ${formatRatioPercent(embeddingIncreaseRate, { clampToUnit: true })}`}
          foot="来源：后端 speaker embedding 距离"
          tone="red"
        />
        {hasCloneConfidence ? (
          <DeltaStatCard
            title="克隆可置信度"
            before={formatRatioPercent(confidenceBefore, { clampToUnit: true })}
            after={formatRatioPercent(confidenceAfter, { clampToUnit: true })}
            delta={`↓ ${formatRatioPercent(confidenceDropRate, { clampToUnit: true })}`}
            foot="来源：校准后的 speaker verification probability model"
            tone="green"
          />
        ) : null}
      </div>
      {!hasCloneConfidence ? (
        <p className="mt-3 text-[11px] leading-5 text-slate-500">克隆置信度需要校准后的 speaker verification probability model，当前未配置，因此不展示。</p>
      ) : null}
    </section>
  )
}

function CloneResultPanel({ cloneEval }: { cloneEval: CloneEval }) {
  const similarityDropRate = optionalNumber(cloneEval.similarityDropRate) ?? computeDropRate(cloneEval.originalSimilarity, cloneEval.protectedSimilarity)

  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <SectionTitle>克隆防护结果</SectionTitle>
      <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3">
        <ScoreBox label="原始克隆相似度" value={formatMetricValue(cloneEval.originalSimilarity, 'number')} />
        <ScoreBox label="保护后克隆相似度" value={formatMetricValue(cloneEval.protectedSimilarity, 'number')} />
        <ScoreBox label="防护下降率" value={formatRatioPercent(similarityDropRate, { clampToUnit: true })} />
      </div>
    </section>
  )
}

function CloneVisualizationPanel({ result, cloneEval }: { result: TaskResult; cloneEval: CloneEval }) {
  const radar = cloneEval.cloneRadar ?? result.speakerFeatureMap?.radar ?? result.charts?.speakerRadar ?? null
  const displayRadar = normalizeCloneRadarForDisplay(radar ?? [], cloneEval)
  const availableRadar = displayRadar.filter((item) => typeof item.value === 'number' && Number.isFinite(item.value))

  return (
    <section className="flex h-full flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex items-center justify-between gap-4">
        <SectionTitle>说话人防护雷达图</SectionTitle>
        <span className="text-[10px] text-slate-500">由后端真实指标动态生成</span>
      </div>
      <div className="mt-5 min-h-[250px] flex-1 overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-3">
        {!displayRadar.length ? (
          <ChartEmptyState text="后端未返回说话人防护雷达数据" />
        ) : availableRadar.length < 3 ? (
          <ChartEmptyState text="后端返回的可用雷达指标不足，至少需要 3 个真实指标" />
        ) : (
          <CloneRadarPreview radar={displayRadar} availableRadar={availableRadar} />
        )}
      </div>
    </section>
  )
}

function ChartEmptyState({ text }: { text: string }) {
  return <div className="grid h-full min-h-[110px] place-items-center text-center text-xs leading-5 text-slate-500">{text}</div>
}

function normalizeCloneRadarForDisplay(radar: RadarPoint[], cloneEval: CloneEval) {
  const hasRealCloneConfidence = optionalNumber(cloneEval.cloneConfidenceDropRate) !== null
  return radar
    .filter((item) => hasRealCloneConfidence || !/置信|confidence/i.test(item.name))
    .map((item) => ({
      ...item,
      name: normalizeCloneRadarName(item.name),
    }))
}

function normalizeCloneRadarName(name: string) {
  if (/直接|direct/i.test(name)) return '直接声纹偏移'
  if (/相似|similar/i.test(name)) return '相似度下降'
  if (/嵌入|embedding|距离/i.test(name)) return '嵌入距离增加'
  if (/保护后|防护|protected/i.test(name)) return '保护后克隆防护'
  if (/置信|confidence/i.test(name)) return '克隆置信度下降'
  return name
}

function CloneRadarPreview({ radar, availableRadar }: { radar: RadarPoint[]; availableRadar: RadarPoint[] }) {
  const width = 510
  const height = 345
  const centerX = width / 2
  const centerY = height / 2
  const radius = 102
  const axisPoints = availableRadar.map((item, index) => {
    const angle = -Math.PI / 2 + (index / availableRadar.length) * Math.PI * 2
    const normalized = Math.max(0, Math.min(100, item.value ?? 0)) / 100
    return {
      item,
      angle,
      axisX: centerX + Math.cos(angle) * radius,
      axisY: centerY + Math.sin(angle) * radius,
      valueX: centerX + Math.cos(angle) * radius * normalized,
      valueY: centerY + Math.sin(angle) * radius * normalized,
      labelX: centerX + Math.cos(angle) * (radius + 51),
      labelY: centerY + Math.sin(angle) * (radius + 36),
    }
  })
  const polygon = axisPoints.map((point) => `${point.valueX.toFixed(1)},${point.valueY.toFixed(1)}`).join(' ')
  const missing = radar.filter((item) => !(typeof item.value === 'number' && Number.isFinite(item.value)))
  const missingNames = missing.map((item) => item.reason ? `${item.name}（${item.reason}）` : item.name).filter(Boolean)

  return (
    <div className="flex h-full flex-col">
      <div className="grid min-h-0 flex-1 place-items-center">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-[345px] w-full max-w-[840px]">
          {[0.25, 0.5, 0.75, 1].map((scale) => (
            <polygon
              key={scale}
              points={axisPoints.map((point) => `${(centerX + Math.cos(point.angle) * radius * scale).toFixed(1)},${(centerY + Math.sin(point.angle) * radius * scale).toFixed(1)}`).join(' ')}
              fill="none"
              stroke="rgba(148,163,184,.16)"
              strokeWidth="1"
            />
          ))}
          {axisPoints.map((point) => (
            <line key={point.item.name} x1={centerX} y1={centerY} x2={point.axisX} y2={point.axisY} stroke="rgba(148,163,184,.18)" strokeWidth="1" />
          ))}
          <polygon points={polygon} fill="rgba(34,211,238,.18)" stroke="#67e8f9" strokeWidth="2" />
          {axisPoints.map((point) => (
            <g key={point.item.name}>
              <circle cx={point.valueX} cy={point.valueY} r="3.5" fill="#fcd34d" />
              <text x={point.labelX} y={point.labelY} textAnchor={point.labelX < centerX - 12 ? 'end' : point.labelX > centerX + 12 ? 'start' : 'middle'} fontSize="13" fontWeight="700" fill="#cbd5e1">
                {point.item.name}
              </text>
              <text x={point.labelX} y={point.labelY + 17} textAnchor={point.labelX < centerX - 12 ? 'end' : point.labelX > centerX + 12 ? 'start' : 'middle'} fontSize="12" fill="#67e8f9">
                {formatRadarScore(point.item.value)}
              </text>
            </g>
          ))}
        </svg>
      </div>
      {missingNames.length ? (
        <div className="overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-4">
          <p className="text-xs leading-5 text-slate-500">部分指标未生成：{missingNames.join('；')}</p>
        </div>
      ) : null}
    </div>
  )
}

function AsrEvalModal({
  model,
  error,
  loading,
  modelOptions,
  onChange,
  onClose,
  onSubmit,
}: {
  model: string
  error?: string
  loading: boolean
  modelOptions: string[]
  onChange: (model: string) => void
  onClose: () => void
  onSubmit: () => void
}) {
  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="ASR 测试表单">
      <div className="ui-card w-full max-w-[460px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-[20px] font-black text-white">ASR 测试</h3>
            <p className="mt-1 text-xs text-slate-500">POST /api/tasks/{'{taskId}'}/asr-eval</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭 ASR 测试表单">
            <X className="h-4 w-4" />
          </button>
        </div>
        <label className="text-sm font-bold text-slate-300">
          ASR 模型
          <select value={model} onChange={(event) => onChange(event.target.value)} className="mt-2 h-10 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950/70 px-3 text-slate-100 outline-none focus:border-cyan-300">
            {modelOptions.map((item) => (
              <option key={item} value={item} className="bg-slate-950 text-slate-100">
                {item}
              </option>
            ))}
          </select>
        </label>
        {error ? <p className="mt-4 rounded-[7px] border border-red-300/20 bg-red-400/10 px-3 py-2 text-sm text-red-100">{error}</p> : null}
        <div className="mt-5 flex justify-end gap-3">
          <button type="button" onClick={onClose} className="h-10 rounded-[7px] border border-cyan-300/14 bg-white/[0.035] px-4 text-sm font-bold text-slate-300">
            取消
          </button>
          <button type="button" onClick={onSubmit} disabled={loading} className="cyan-button inline-flex h-10 min-w-[116px] items-center justify-center gap-2 rounded-[7px] px-4 text-sm font-black disabled:opacity-60">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <TestTube2 className="h-4 w-4" />}
            开始测试
          </button>
        </div>
      </div>
    </div>
  )
}

function CloneVoiceModal({
  form,
  error,
  loading,
  modelOptions,
  languageOptions,
  speedOptions,
  onChange,
  onClose,
  onSubmit,
}: {
  form: CloneVoiceRequest
  error?: string
  loading: boolean
  modelOptions: BackendSelectOption[]
  languageOptions: string[]
  speedOptions: number[]
  onChange: (form: CloneVoiceRequest) => void
  onClose: () => void
  onSubmit: () => void
}) {
  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="语音克隆测试表单">
      <div className="ui-card w-full max-w-[620px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-[20px] font-black text-white">语音克隆测试</h3>
            <p className="mt-1 text-xs text-slate-500">POST /api/tasks/{'{taskId}'}/clone-voice</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭语音克隆测试表单">
            <X className="h-4 w-4" />
          </button>
        </div>
        <label className="block text-sm font-bold text-slate-300">
          克隆文本
          <textarea
            value={form.text}
            onChange={(event) => onChange({ ...form, text: event.target.value })}
            className="mt-2 min-h-[126px] w-full resize-none rounded-[7px] border border-cyan-300/14 bg-slate-950/70 px-3 py-3 text-sm leading-6 text-slate-100 outline-none transition focus:border-cyan-300"
          />
        </label>
        <div className="mt-4 grid grid-cols-[1fr_150px_120px] gap-3">
          <label className="text-sm font-bold text-slate-300">
            克隆模型
            <select
              value={form.model}
              onChange={(event) => {
                const model = event.target.value
                const selected = modelOptions.find((item) => item.value === model)
                const languages = selected?.languages?.length ? selected.languages : languageOptions
                onChange({ ...form, model, language: languages.includes(form.language ?? '') ? form.language : languages[0] })
              }}
              className="mt-2 h-10 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950/70 px-3 text-slate-100 outline-none focus:border-cyan-300"
            >
              {modelOptions.map((item) => (
                <option key={item.value} value={item.value} className="bg-slate-950 text-slate-100">
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm font-bold text-slate-300">
            语言
            <select value={form.language ?? 'auto'} onChange={(event) => onChange({ ...form, language: event.target.value })} className="mt-2 h-10 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950/70 px-3 text-slate-100 outline-none focus:border-cyan-300">
              {languageOptions.map((item) => (
                <option key={item} value={item} className="bg-slate-950 text-slate-100">
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm font-bold text-slate-300">
            语速
            <select value={String(form.speed ?? 1)} onChange={(event) => onChange({ ...form, speed: Number(event.target.value) })} className="mt-2 h-10 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950/70 px-3 text-slate-100 outline-none focus:border-cyan-300">
              {speedOptions.map((item) => (
                <option key={item} value={item} className="bg-slate-950 text-slate-100">
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className="mt-4 block text-sm font-bold text-slate-300">
          模型补充参数
          <input value={form.speakerPrompt ?? ''} onChange={(event) => onChange({ ...form, speakerPrompt: event.target.value })} className="mt-2 h-10 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950/70 px-3 text-slate-100 outline-none focus:border-cyan-300" placeholder="可选：speaker prompt / speaker id / voice id" />
        </label>
        {error ? <p className="mt-4 rounded-[7px] border border-red-300/20 bg-red-400/10 px-3 py-2 text-sm text-red-100">{error}</p> : null}
        <div className="mt-5 flex justify-end gap-3">
          <button type="button" onClick={onClose} className="h-10 rounded-[7px] border border-cyan-300/14 bg-white/[0.035] px-4 text-sm font-bold text-slate-300">
            取消
          </button>
          <button type="button" onClick={onSubmit} disabled={loading} className="cyan-button inline-flex h-10 min-w-[128px] items-center justify-center gap-2 rounded-[7px] px-4 text-sm font-black disabled:opacity-60">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <TestTube2 className="h-4 w-4" />}
            开始测试
          </button>
        </div>
      </div>
    </div>
  )
}

function AudioCard({
  title,
  audio,
  color,
  green,
  onPlayRequest,
}: {
  title: string
  audio: AudioFileMeta
  color: string
  green?: boolean
  onPlayRequest?: () => Promise<string | undefined>
}) {
  const src = getAudioSource(audio)
  const duration = getAudioDuration(audio)

  return (
    <div className={cn('flex h-[252px] flex-col rounded-[9px] border p-5', green ? 'border-emerald-400/18 bg-emerald-400/8' : 'border-cyan-300/14 bg-[#07192d]/80')}>
      <p className="flex items-center gap-2 whitespace-nowrap text-sm font-black text-slate-200">
        {green ? <ShieldCheck className="h-4 w-4 text-emerald-300" /> : <Volume2 className="h-4 w-4 text-sky-300" />}
        {title}
      </p>
      <p className="ml-6 mt-0.5 flex min-w-0 text-xs text-slate-400">
        <span className="truncate">{audio.filename.replace(/\.[^.]+$/, '')}</span>
        <span className="shrink-0">{audio.filename.match(/\.[^.]+$/)?.[0] ?? ''}</span>
      </p>
      <TinyWave color={color} className="h-[58px]" />
      <div className="mt-4 h-1 rounded-full bg-slate-700">
        <div className="h-full w-[35%] rounded-full" style={{ background: color }} />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[10px] text-slate-400">
        <span>00:00</span>
        <span>{formatDurationSeconds(duration)}</span>
      </div>
      <div className="mt-auto">
        <AudioPlayer
          src={src}
          title={title}
          filename={audio.filename}
          disabledReason={green ? '点击播放时将从后端下载保护音频' : '暂无原始音频 URL'}
          downloadable={Boolean(src)}
          downloadFilename={audio.filename}
          onPlayRequest={onPlayRequest}
        />
      </div>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 border-t border-cyan-300/10 pt-3 pb-1 text-[12px] text-slate-400">
        <span>时长 {duration ? `${duration.toFixed(2)}s` : '待解析'}</span>
        <span>采样率 {audio.sampleRate ? `${audio.sampleRate / 1000}kHz` : '待解析'}</span>
        <span>声道 {audio.channels ?? '待解析'}</span>
        <span>格式 {audio.format}</span>
        <span>大小 {formatFileSize(audio.sizeBytes)}</span>
      </div>
    </div>
  )
}

function TextBox({ title, text, foot, content }: { title: string; text: string; foot: string; content?: ReactNode }) {
  return (
    <div className="flex h-[226px] flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/18 p-4">
      <h3 className="mb-3 whitespace-nowrap text-sm font-bold text-slate-300">{title}</h3>
      <div className="min-h-[122px] overflow-y-auto rounded-[7px] border border-cyan-300/8 bg-slate-950/22 p-4 text-[13px] leading-6 text-slate-200">
        {content ?? text}
      </div>
      <p className="mt-auto truncate pt-4 text-[12px] text-slate-500">{foot}</p>
    </div>
  )
}

function chooseEditLevel(original: string, next: string): EditLevel {
  const hasCjk = /[\u3400-\u9fff]/.test(`${original}${next}`)
  if (hasCjk) return 'char'
  return /\s/.test(original.trim()) || /\s/.test(next.trim()) ? 'word' : 'char'
}

function tokenizeText(text: string, level: EditLevel) {
  if (level === 'word') return text.trim().split(/\s+/).filter(Boolean)
  return Array.from(text)
}

function computeEditMetrics(originalText: string, protectedText: string, level: EditLevel): EditMetrics | null {
  const original = tokenizeText(originalText, level)
  const next = tokenizeText(protectedText, level)
  if (original.length === 0) return null
  const dp = Array.from({ length: original.length + 1 }, () => Array<number>(next.length + 1).fill(0))
  for (let i = 0; i <= original.length; i += 1) dp[i][0] = i
  for (let j = 0; j <= next.length; j += 1) dp[0][j] = j
  for (let i = 1; i <= original.length; i += 1) {
    for (let j = 1; j <= next.length; j += 1) {
      if (original[i - 1] === next[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1]
      } else {
        dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
      }
    }
  }

  let i = original.length
  let j = next.length
  let substitutions = 0
  let insertions = 0
  let deletions = 0
  const ops: DiffOp[] = []

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && original[i - 1] === next[j - 1]) {
      ops.unshift({ type: 'equal', text: original[i - 1] })
      i -= 1
      j -= 1
    } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
      substitutions += 1
      ops.unshift({ type: 'replace', from: original[i - 1], to: next[j - 1] })
      i -= 1
      j -= 1
    } else if (j > 0 && dp[i][j] === dp[i][j - 1] + 1) {
      insertions += 1
      ops.unshift({ type: 'insert', text: next[j - 1] })
      j -= 1
    } else {
      deletions += 1
      ops.unshift({ type: 'delete', text: original[i - 1] })
      i -= 1
    }
  }

  const base = original.length
  const totalErrors = substitutions + insertions + deletions
  const errorBase = Math.max(totalErrors, 1)
  return {
    level,
    werOrCer: totalErrors / base,
    substituteRate: substitutions / base,
    insertRate: insertions / base,
    deleteRate: deletions / base,
    referenceLength: base,
    substitutions,
    insertions,
    deletions,
    totalErrors,
    errorShares: {
      substituteShare: totalErrors ? substitutions / errorBase : 0,
      insertShare: totalErrors ? insertions / errorBase : 0,
      deleteShare: totalErrors ? deletions / errorBase : 0,
    },
    diffOps: ops,
  }
}

function asrErrorShares(asrEval: AsrEval, editStats: EditMetrics | null) {
  const direct = asrEval.errorShares
  if (direct && [direct.substituteShare, direct.insertShare, direct.deleteShare].some((value) => optionalNumber(value) !== null)) {
    return {
      substituteShare: optionalNumber(direct.substituteShare) ?? 0,
      insertShare: optionalNumber(direct.insertShare) ?? 0,
      deleteShare: optionalNumber(direct.deleteShare) ?? 0,
    }
  }
  const counts = asrEval.editCounts
  if (counts && optionalNumber(counts.totalErrors) !== null) {
    const total = Math.max(optionalNumber(counts.totalErrors) ?? 0, 1)
    return {
      substituteShare: counts.totalErrors ? counts.substitutions / total : 0,
      insertShare: counts.totalErrors ? counts.insertions / total : 0,
      deleteShare: counts.totalErrors ? counts.deletions / total : 0,
    }
  }
  return editStats?.errorShares ?? null
}

function renderDiffOps(diffOps: DiffOp[]): ReactNode[] {
  const joiner = diffOps.some((op) => ('text' in op ? /\s/.test(op.text) : /\s/.test(`${op.from}${op.to}`))) ? ' ' : ''
  const nodes: ReactNode[] = []
  diffOps.forEach((op, index) => {
    const spacer = index === diffOps.length - 1 ? '' : joiner
    if (op.type === 'equal') {
      nodes.push(`${op.text}${spacer}`)
      return
    }
    if (op.type === 'insert') {
      nodes.push(
        <span key={`ins-${index}`} className="text-red-300">
          {op.text}
          {spacer}
        </span>,
      )
      return
    }
    if (op.type === 'delete') {
      nodes.push(
        <span key={`del-${index}`} className="text-emerald-300 line-through decoration-emerald-300/70">
          {op.text}
          {spacer}
        </span>,
      )
      return
    }
    if ('from' in op) {
      nodes.push(
        <span key={`replace-del-${index}`} className="text-emerald-300 line-through decoration-emerald-300/70">
          {op.from}
        </span>,
        <span key={`replace-ins-${index}`} className="text-red-300">
          {joiner}
          {op.to}
          {spacer}
        </span>,
      )
    }
  })
  return nodes
}

function ScoreBox({ label, value, red, compact, foot }: { label: string; value: string; red?: boolean; compact?: boolean; foot?: string }) {
  return (
    <div className={cn('rounded-[9px] border border-cyan-300/12 bg-slate-950/16 text-center', compact ? 'min-h-[64px] p-2.5' : 'min-h-[82px] p-3')}>
      <p className="mx-auto max-w-full text-[11px] leading-4 text-slate-400">{label}</p>
      <div className="mt-2 grid justify-items-center">
        <span className={cn(compact ? 'text-[19px]' : 'text-[24px]', 'break-words font-black leading-none', red ? 'text-red-300' : 'text-cyan-300')}>
          {value}
        </span>
      </div>
      {foot ? <p className="mt-1 line-clamp-2 text-[10px] leading-4 text-slate-500" title={foot}>{foot}</p> : null}
    </div>
  )
}

function DeltaStatCard({ title, before, after, delta, foot, tone }: { title: string; before: string; after: string; delta: string; foot: string; tone: 'green' | 'red' }) {
  return (
    <div className="min-h-[180px] rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-4">
      <h3 className="min-h-[40px] text-[13px] font-bold leading-5 text-slate-300">{title}</h3>
      <div className="mt-3 grid grid-cols-[minmax(0,1fr)_28px_minmax(0,1fr)] items-center text-center text-[20px]">
        <span className="min-w-0 break-words text-slate-200">{before}</span>
        <span className="text-slate-400">→</span>
        <span className="min-w-0 break-words text-emerald-300">{after}</span>
      </div>
      <div className={cn('mt-2 rounded-[5px] py-2 text-center font-black', tone === 'green' ? 'bg-emerald-400/14 text-emerald-300' : 'bg-red-400/12 text-red-300')}>{delta}</div>
      <p className="mt-2 text-[11px] leading-4 text-slate-500">{foot}</p>
    </div>
  )
}

function QualityPanel({ result, embedded }: { result: TaskResult; embedded?: boolean }) {
  const snr = optionalNumber(result.protectionQuality?.snr) ?? optionalNumber(result.quality.snr)
  const pesq = optionalNumber(result.protectionQuality?.pesq) ?? optionalNumber(result.quality.pesq)
  const stoi = optionalNumber(result.protectionQuality?.stoi)
  const backendMos = optionalNumber(result.protectionQuality?.mos)
  const manualMosKey = `manual-mos:${result.taskId || 'current'}`
  const [manualMos, setManualMos] = useState<number | null>(null)
  const [editingMos, setEditingMos] = useState(false)
  const [mosDraft, setMosDraft] = useState('')
  const mos = backendMos ?? manualMos
  const missingReasons = [
    pesq === null ? ['PESQ', metricReason(result, ['protectionQuality.pesq'])] : null,
    stoi === null ? ['STOI', metricReason(result, ['protectionQuality.stoi'])] : null,
  ].filter((item): item is [string, string] => Boolean(item?.[1]))

  useEffect(() => {
    const saved = window.localStorage.getItem(manualMosKey)
    if (saved === null) {
      setManualMos(null)
      return
    }
    const value = Number(saved)
    setManualMos(Number.isFinite(value) ? clamp(value, 1, 5) : null)
  }, [manualMosKey])

  const startMosEdit = () => {
    if (backendMos !== null) return
    setMosDraft(manualMos === null ? '' : String(manualMos))
    setEditingMos(true)
  }

  const commitMosEdit = () => {
    const trimmed = mosDraft.trim()
    if (!trimmed) {
      window.localStorage.removeItem(manualMosKey)
      setManualMos(null)
      setEditingMos(false)
      return
    }
    const value = Number(trimmed)
    if (!Number.isFinite(value)) {
      setEditingMos(false)
      return
    }
    const normalized = Math.round(clamp(value, 1, 5) * 10) / 10
    window.localStorage.setItem(manualMosKey, String(normalized))
    setManualMos(normalized)
    setEditingMos(false)
  }

  return (
    <section className={cn(embedded ? 'mt-5' : 'ui-card p-5')}>
      <SectionTitle>感知质量评估</SectionTitle>
      <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(132px,1fr))] gap-3">
        <QualityMetric label="SNR（信噪比）" value={formatMetricValue(snr, 'db')} tag={snr === null ? '未生成' : 'computed'} tone="green" />
        <QualityMetric label="PESQ" value={formatMetricValue(pesq, 'number')} tag={pesq === null ? '未生成' : 'perception'} tone="blue" />
        <QualityMetric label="STOI" value={formatMetricValue(stoi, 'number')} tag={stoi === null ? '未生成' : 'perception'} tone="blue" />
        <QualityMetric
          label="MOS（人工）"
          value={
            editingMos ? (
              <input
                autoFocus
                className="h-7 w-20 rounded-[6px] border border-orange-300/28 bg-slate-950/80 px-2 text-center text-[18px] font-black leading-none text-orange-200 outline-none focus:border-orange-200"
                inputMode="decimal"
                max={5}
                min={1}
                onBlur={commitMosEdit}
                onChange={(event) => setMosDraft(event.target.value)}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') commitMosEdit()
                  if (event.key === 'Escape') setEditingMos(false)
                }}
                step={0.1}
                type="number"
                value={mosDraft}
              />
            ) : mos === null ? (
              '自行评价'
            ) : (
              formatMetricValue(mos, 'number')
            )
          }
          onClick={backendMos === null ? startMosEdit : undefined}
          tag={backendMos === null ? (manualMos === null ? '点击输入' : '人工反馈') : 'perception'}
          title={backendMos === null ? '点击输入 1-5 分人工 MOS' : undefined}
          tone="orange"
        />
      </div>
      {missingReasons.length ? (
        <p className="mt-3 text-xs leading-5 text-slate-500">
          未生成原因：{missingReasons.map(([name, reason]) => `${name}: ${reason}`).join('；')}
        </p>
      ) : null}
    </section>
  )
}

function PsychoacousticPanel({ result }: { result: TaskResult }) {
  return (
    <section className="flex min-h-[296px] flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex items-center justify-between gap-4">
        <SectionTitle>心理声学阈值分析（关键频段）</SectionTitle>
        <div className="flex shrink-0 gap-4 text-[10px] text-slate-400">
          <span className="flex items-center gap-1.5 text-cyan-200">
            <span className="h-0 w-5 border-t-2 border-dashed border-cyan-300" />
            掩蔽阈值
          </span>
          <span className="flex items-center gap-1.5 text-amber-200">
            <span className="h-0 w-5 border-t-2 border-amber-300" />
            防护扰动谱
          </span>
        </div>
      </div>
      <div className="mt-5 min-h-0 flex-1 overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/16 px-4 py-3">
        <LineChart result={result} large />
      </div>
    </section>
  )
}

function QualityMetric({ label, value, tag, tone, onClick, title }: { label: string; value: ReactNode; tag: string; tone: 'green' | 'blue' | 'orange'; onClick?: () => void; title?: string }) {
  return (
    <div
      className={cn('h-[86px] rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-3 text-center', onClick && 'cursor-pointer transition hover:border-orange-300/28 hover:bg-orange-300/[0.04]')}
      onClick={onClick}
      onKeyDown={(event) => {
        if (!onClick) return
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onClick()
        }
      }}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      title={title}
    >
      <p className="whitespace-nowrap text-[11px] text-slate-400">{label}</p>
      <div className={cn('mt-1 flex h-6 items-center justify-center text-[20px] font-black leading-none', tone === 'green' && 'text-emerald-300', tone === 'blue' && 'text-cyan-300', tone === 'orange' && 'text-orange-300')}>{value}</div>
      <span className={cn('mt-1.5 inline-block rounded px-3 py-0.5 text-[11px] font-bold', tone === 'green' && 'bg-emerald-400/14 text-emerald-300', tone === 'blue' && 'bg-cyan-400/14 text-cyan-300', tone === 'orange' && 'bg-orange-400/14 text-orange-300')}>{tag}</span>
    </div>
  )
}

type LossDisplayKey = 'Lid' | 'Lsem' | 'Lpsy' | 'L2'
type LossDefinition = { key: LossDisplayKey; legacyKey?: 'Lfeat'; formula: string; altFormula?: string; label: string; description: string; colorClass: string }

const lossDefinitions: LossDefinition[] = [
  {
    key: 'Lid',
    legacyKey: 'Lfeat',
    formula: 'L_{\\mathrm{id}}',
    label: 'Identity Loss',
    description: '声音身份损失',
    colorClass: 'bg-cyan-300',
  },
  { key: 'Lsem', formula: 'L_{\\mathrm{sem}}', label: 'Semantic Loss', description: '语义损失', colorClass: 'bg-emerald-300' },
  { key: 'Lpsy', formula: 'L_{\\mathrm{psy}}', label: 'Psychoacoustic Loss', description: '心理声学损失，量级可能较大', colorClass: 'bg-amber-300' },
  { key: 'L2', formula: 'L_2', altFormula: '\\lVert\\delta\\rVert_2', label: 'L2 Constraint', description: '扰动范数约束', colorClass: 'bg-violet-300' },
]

function TrendPanel({ result, embedded }: { result: TaskResult; embedded?: boolean }) {
  const trend = downsampleTrace(result.optimizationTrace ?? result.generation?.optimizationTrace ?? result.charts.optimizationTrend)
  const lossFinal = result.lossFinal ?? result.generation?.lossFinal ?? finalLossFromTrend(trend)
  const missingLosses = lossDefinitions.filter((loss) => trend.length > 0 && trend.every((point) => lossPointValue(point, loss) === null))
  const steps = result.generation?.steps ?? lastStep(trend)
  const avgIterationSec = optionalNumber(result.averageStepSec) ?? averageStepSecFromTrace(trend) ?? (typeof result.elapsedSec === 'number' && steps && steps > 0 ? result.elapsedSec / steps : null)

  return (
    <section className={cn('flex h-full min-h-[380px] flex-col overflow-hidden', embedded ? 'rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-5' : 'ui-card p-7')}>
      <div className="flex items-start justify-between gap-8">
        <SectionTitle>优化损失趋势</SectionTitle>
        <div className="shrink-0 rounded-[7px] border border-cyan-300/12 bg-slate-950/20 px-6 py-3.5 text-right">
          <p className="text-[10px] text-slate-500">平均每次迭代耗时</p>
          <p className="text-[15px] font-black text-cyan-200">{avgIterationSec === null ? '未生成' : `${avgIterationSec.toFixed(3)} s / step`}</p>
        </div>
      </div>
      <div className="mt-7 grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_280px] gap-6 max-lg:grid-cols-1">
        <div className="flex min-h-0 flex-col rounded-[7px] border border-cyan-300/12 bg-slate-950/18 p-6">
          <div className="mb-5 flex flex-wrap items-center gap-x-8 gap-y-2 text-[11px] text-slate-400">
            {lossDefinitions.map((loss) => (
              <span key={loss.key} className="inline-flex items-center gap-1.5">
                <span className={cn('h-2.5 w-2.5 rounded-full', loss.colorClass)} />
                <MathText formula={loss.formula} />
                {loss.altFormula ? <MathText formula={loss.altFormula} /> : null}
              </span>
            ))}
          </div>
          {trend.length > 0 ? (
            <div className="min-h-0 flex-1">
              <TrendChart data={trend} />
            </div>
          ) : (
            <div className="grid min-h-[210px] flex-1 place-items-center rounded-[6px] border border-dashed border-cyan-300/14 bg-slate-950/16 px-5 text-center text-[12px] leading-5 text-slate-400">
              后端未记录逐步优化损失，当前仅可在详细数据中查看最终 loss。
            </div>
          )}
        </div>
        <div className="grid content-start gap-4 overflow-y-auto pr-1">
          {lossDefinitions.map((loss) => (
            <div key={loss.key} className="rounded-[7px] border border-cyan-300/12 bg-slate-950/18 px-5 py-4">
              <div className="flex items-center justify-between gap-3">
                <p className="flex items-center gap-2 text-[12px] font-bold text-slate-200">
                  <MathText formula={loss.formula} className="text-cyan-100" />
                  {loss.altFormula ? <MathText formula={loss.altFormula} className="text-cyan-100" /> : null}
                  <span className="text-slate-500">{loss.label}</span>
                </p>
                <p className="text-[13px] font-black text-white">{formatLossNumber(lossFinalValue(lossFinal, loss))}</p>
              </div>
              <p className="mt-1 text-[10px] text-slate-500">{loss.description}</p>
            </div>
          ))}
          <div className="rounded-[7px] border border-cyan-300/12 bg-slate-950/18 px-5 py-4">
            <div className="flex items-center justify-between gap-3">
              <p className="text-[12px] font-bold text-slate-300">total loss</p>
              <p className="text-[13px] font-black text-white">{formatLossNumber(lossFinal?.total)}</p>
            </div>
          </div>
        </div>
      </div>
      {missingLosses.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
          {missingLosses.map((loss) => (
            <span key={loss.key}>
              <MathText formula={loss.formula} className="align-[-1px]" />：后端未返回
            </span>
          ))}
        </div>
      ) : null}
    </section>
  )
}

function InsightPanel({ title, items }: { title: string; items: string[] }) {
  return (
    <section className="flex h-full min-h-[260px] flex-col overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-5">
      <SectionTitle>
        {title} <span className="text-sm font-normal text-slate-500">（自动生成）</span>
      </SectionTitle>
      <div className="mt-6 grid max-h-[520px] flex-1 grid-cols-1 content-start gap-4 overflow-y-auto rounded-[7px] border border-cyan-300/10 bg-slate-950/12 p-5 pr-2 text-[14px] leading-7 text-slate-200">
        {items.map((item) => (
          <p key={item} className="flex min-w-0 gap-3">
            <CheckCircle2 className="mt-1.5 h-4 w-4 shrink-0 text-emerald-300" />
            <span>{item}</span>
          </p>
        ))}
      </div>
      <p className="mt-2 text-right text-[11px] text-slate-500">以上分析仅基于前端可见字段，不调用后端 AI。</p>
    </section>
  )
}

function taskInfoRows(result: TaskResult): Array<[string, string]> {
  return [
    ['任务 ID', result.taskId],
    ['提交时间', result.submittedAt ?? result.createdAt ?? result.originalAudio.uploadedAt ?? '-'],
    ['完成时间', result.completedAt ?? '-'],
    ['处理耗时', typeof result.elapsedSec === 'number' ? formatElapsed(result.elapsedSec) : '-'],
    ['输入来源', result.inputSource ?? '手动上传'],
    ['音频时长', formatDurationSeconds(getAudioDuration(result.originalAudio))],
    ['语言类型', result.language ?? '未标注'],
    ['处理模型', result.processingModel ?? result.asrModel ?? modeText[result.mode] ?? result.mode],
    ['优化目标', result.optimizationTarget ?? result.mode],
    ['防护模式', modeText[result.mode] ?? result.mode],
  ]
}

function TaskInfoModal({ result, onClose }: { result: TaskResult; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="任务信息">
      <div className="ui-card w-full max-w-[560px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div>
            <h3 className="text-[20px] font-black text-white">任务信息</h3>
            <p className="mt-1 text-xs text-slate-500">GET /api/tasks/{'{taskId}'}/details</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭任务信息">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="max-h-[520px] overflow-y-auto rounded-[9px] border border-cyan-300/12 bg-slate-950/18 p-4 pr-2">
          {taskInfoRows(result).map(([label, value]) => (
            <p key={label} className="mb-3 grid grid-cols-[88px_minmax(0,1fr)] text-[13px] leading-5 last:mb-0">
              <span className="text-slate-500">{label}</span>
              <span className="min-w-0 break-words font-semibold text-slate-300">{value}</span>
            </p>
          ))}
        </div>
      </div>
    </div>
  )
}

function DownloadModal({ result, onClose }: { result: TaskResult; onClose: () => void }) {
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)

  const runDownload = async (kind: 'audio' | 'report' | 'zip') => {
    try {
      let blob: Blob
      let filename: string

      if (kind === 'audio') {
        const file = await downloadProtectedAudio(result.taskId)
        blob = file.blob
        filename = file.filename
      } else if (kind === 'report') {
        blob = await exportReport(result.taskId)
        filename = `${result.taskId}-report.pdf`
      } else {
        blob = await downloadEvidenceZip(result.taskId)
        filename = `${result.taskId}-evidence.zip`
      }

      downloadBlob(blob, filename)
      pushToast({ kind: 'success', title: '下载已开始', description: filename })
    } catch (error) {
      pushToast({ kind: 'error', title: '导出暂不可用', description: error instanceof Error ? error.message : '请稍后重试。' })
    }
  }

  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="下载与导出">
      <div className="ui-card w-full max-w-[520px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-5 flex items-center justify-between gap-4">
          <div>
            <h3 className="text-[20px] font-black text-white">{result.verdict || '防护结果已生成'}</h3>
            <p className="mt-1 text-xs text-slate-500">点击此处下载</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭下载与导出">
            <X className="h-4 w-4" />
          </button>
        </div>
        <button onClick={() => void runDownload('audio')} className="cyan-button flex h-12 w-full items-center justify-center gap-2 rounded-[8px] text-[16px] font-black">
          <Download className="h-4 w-4" />
          下载保护音频
        </button>
        {['导出评估报告（PDF）', '下载完整证据链（ZIP）', '重新执行任务'].map((item, index) => (
          <button
            key={item}
            onClick={() => {
              if (index === 0) void runDownload('report')
              if (index === 1) void runDownload('zip')
              if (index === 2) navigate('/workspace')
            }}
            className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-[8px] border border-cyan-300/12 bg-white/[0.035] text-[16px] font-bold text-slate-300"
          >
            {index === 0 ? <FileText className="h-4 w-4" /> : index === 1 ? <FileArchive className="h-4 w-4" /> : <RefreshCw className="h-4 w-4" />}
            {item}
          </button>
        ))}
      </div>
    </div>
  )
}

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '未生成'
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)}MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${bytes}B`
}

function formatMetricValue(value: unknown, type: 'percent' | 'db' | 'seconds' | 'loss' | 'bytes' | 'number') {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  if (type === 'percent') return `${(numberValue <= 1 ? numberValue * 100 : numberValue).toFixed(1)}%`
  if (type === 'db') return `${numberValue.toFixed(1)} dB`
  if (type === 'seconds') return `${numberValue.toFixed(3)} s`
  if (type === 'loss') return formatLossNumber(numberValue)
  if (type === 'bytes') return formatFileSize(numberValue)
  return numberValue.toFixed(3).replace(/\.?0+$/, '')
}

function formatRatioPercent(value: unknown, options?: { clampToUnit?: boolean }) {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  const normalized = options?.clampToUnit ? clamp(numberValue, 0, 1) : numberValue
  return `${(normalized * 100).toFixed(1)}%`
}

function formatRadarScore(value: unknown) {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  return `${clamp(numberValue, 0, 100).toFixed(1)} 分`
}

function formatSpeakerEvalModel(value: unknown) {
  const label = typeof value === 'string' && value.trim() ? value.trim() : 'ECAPA-TDNN'
  if (label === 'speechbrain/spkrec-ecapa-voxceleb' || /spkrec-ecapa-voxceleb/i.test(label)) return 'ECAPA-TDNN'
  return label
}

function optionalNumber(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function metricSource(result: TaskResult, keys: string[]) {
  for (const key of keys) {
    const source = result.metricSources?.[key]
    if (source) return source
  }
  return undefined
}

function metricReason(result: TaskResult, keys: string[]) {
  const reason = metricSource(result, keys)?.reason
  return reason ? shortMetricReason(reason) : ''
}

function shortMetricReason(reason: string) {
  if (/torchcodec|libtorchcodec|FFmpeg/i.test(reason)) return '语义 tokenizer/encoder 依赖 torchcodec/FFmpeg 未正确加载'
  if (/local cache|Hub|connection|Internet|from_pretrained|huggingface/i.test(reason)) {
    if (/semantic|encoder|hubert|whisper|tokenizer|s3/i.test(reason)) return '语义编码器模型加载失败'
    if (/speaker|ecapa|speechbrain|spkrec/i.test(reason)) return '说话人模型未在本地缓存，且当前无法从 Hub 加载'
    return '模型未在本地缓存，且当前无法从 Hub 加载'
  }
  const pesqSampleRate = /PESQ supports 8000 or 16000 Hz, got (\d+)/i.exec(reason)
  if (pesqSampleRate) return `PESQ 仅支持 8k/16k，当前 ${pesqSampleRate[1]} Hz`
  if (/confidence calibrator/i.test(reason) || /calibrated clone confidence/i.test(reason)) return '未配置克隆置信度校准模型'
  return reason.split('\n')[0].trim()
}

function formatLossNumber(value: unknown) {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  const abs = Math.abs(numberValue)
  if (abs > 0 && (abs < 0.001 || abs >= 10000)) return numberValue.toExponential(3)
  return numberValue.toFixed(6).replace(/\.?0+$/, '')
}

function finalLossFromTrend(points: LossTrendPoint[]): LossFinal | undefined {
  const last = points.at(-1)
  if (!last) return undefined
  return {
    Lid: last.Lid ?? last.Lfeat,
    Lfeat: last.Lfeat,
    Lsem: last.Lsem,
    Lpsy: last.Lpsy,
    L2: last.L2,
    total: last.total,
  }
}

function lossPointValue(point: LossTrendPoint, loss: { key: LossDisplayKey; legacyKey?: 'Lfeat' }) {
  return optionalNumber(point[loss.key]) ?? (loss.legacyKey ? optionalNumber(point[loss.legacyKey]) : null)
}

function lossFinalValue(lossFinal: LossFinal | null | undefined, loss: { key: LossDisplayKey; legacyKey?: 'Lfeat' }) {
  if (!lossFinal) return null
  return optionalNumber(lossFinal[loss.key]) ?? (loss.legacyKey ? optionalNumber(lossFinal[loss.legacyKey]) : null)
}

function lastStep(points: LossTrendPoint[]) {
  const step = optionalNumber(points.at(-1)?.step)
  return step && step > 0 ? step : null
}

function computeRateChange(before: number | null | undefined, after: number | null | undefined) {
  const beforeValue = optionalNumber(before)
  const afterValue = optionalNumber(after)
  if (beforeValue === null || afterValue === null) return null
  return (afterValue - beforeValue) / Math.max(Math.abs(beforeValue), 1e-8)
}

function computeDropRate(before: number | null | undefined, after: number | null | undefined) {
  const beforeValue = optionalNumber(before)
  const afterValue = optionalNumber(after)
  if (beforeValue === null || afterValue === null) return null
  return (beforeValue - afterValue) / Math.max(beforeValue, 1e-8)
}

function computeEpsilonUsageRate(perturbation: TaskResult['perturbation']) {
  if (!perturbation) return null
  const epsilon = optionalNumber(perturbation.epsilon)
  if (epsilon === null || epsilon <= 0) return null
  const epsilonNorm = perturbation.epsilonNorm
  if (epsilonNorm === 'linf') {
    const linfNorm = optionalNumber(perturbation.linfNorm)
    return linfNorm === null ? null : linfNorm / epsilon
  }
  if (epsilonNorm === 'l2') {
    const l2Norm = optionalNumber(perturbation.l2Norm)
    return l2Norm === null ? null : l2Norm / epsilon
  }
  return null
}

function downsampleTrace(trace: LossTrendPoint[] | null | undefined, maxPoints = 80) {
  const points = trace ?? []
  if (points.length <= maxPoints) return points
  return Array.from({ length: maxPoints }, (_, index) => points[Math.round((index / Math.max(1, maxPoints - 1)) * (points.length - 1))])
}

function averageStepSecFromTrace(trace: LossTrendPoint[]) {
  const values = trace.map((point) => optionalNumber(point.stepElapsedSec)).filter((value): value is number => value !== null)
  if (values.length === 0) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function mean(values: number[]) {
  if (values.length === 0) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function traceMetricValue(point: LossTrendPoint, key: LossDisplayKey | 'total') {
  if (key === 'Lid') return optionalNumber(point.Lid) ?? optionalNumber(point.Lfeat)
  return optionalNumber(point[key])
}

function trendDirection(points: LossTrendPoint[] | null | undefined, key: LossDisplayKey | 'total'): 'increasing' | 'decreasing' | 'stable' | 'unknown' {
  const values = (points ?? []).map((point) => traceMetricValue(point, key)).filter((value): value is number => value !== null)
  if (values.length < 5) return 'unknown'
  const segmentSize = Math.max(1, Math.floor(values.length * 0.2))
  const startMean = mean(values.slice(0, segmentSize))
  const endMean = mean(values.slice(-segmentSize))
  if (startMean === null || endMean === null) return 'unknown'
  const relative = (endMean - startMean) / Math.max(Math.abs(startMean), 1e-8)
  if (relative > 0.1) return 'increasing'
  if (relative < -0.1) return 'decreasing'
  return 'stable'
}

function lossChanged(points: LossTrendPoint[], key: LossDisplayKey | 'total') {
  const direction = trendDirection(points, key)
  return direction === 'increasing' || direction === 'decreasing'
}

function generateProtectionInsights(result: TaskResult) {
  const perturbation = result.perturbation
  const quality = result.protectionQuality ?? result.quality
  const psycho = result.psychoacoustic
  const trace = result.optimizationTrace ?? result.generation?.optimizationTrace ?? result.charts?.optimizationTrend ?? []
  const lossFinal = result.lossFinal ?? result.generation?.lossFinal ?? finalLossFromTrend(trace)
  const snr = optionalNumber(perturbation?.snr) ?? optionalNumber(quality?.snr)
  const pesq = optionalNumber(quality?.pesq)
  const stoi = optionalNumber('stoi' in (quality ?? {}) ? (quality as { stoi?: number | null }).stoi : null)
  const mos = optionalNumber('mos' in (quality ?? {}) ? (quality as { mos?: number | null }).mos : null)
  const l2Norm = optionalNumber(perturbation?.l2Norm)
  const epsilonUsageRate = optionalNumber(perturbation?.epsilonUsageRate)
  const clippingRate = optionalNumber(perturbation?.clippingRate)
  const overMaskRate = optionalNumber(psycho?.overMaskRate)
  const lPsy = optionalNumber(psycho?.lPsy) ?? optionalNumber(lossFinal?.Lpsy)
  const lid = optionalNumber(lossFinal?.Lid) ?? optionalNumber(lossFinal?.Lfeat)
  const lsem = optionalNumber(lossFinal?.Lsem)
  const l2 = optionalNumber(lossFinal?.L2)
  const total = optionalNumber(lossFinal?.total)
  const lPsyTrend = trendDirection(trace, 'Lpsy')
  const lidTrend = trendDirection(trace, 'Lid')
  const l2Trend = trendDirection(trace, 'L2')
  const items: string[] = []
  const missing: string[] = []

  if (l2Norm !== null) {
    const budget = epsilonUsageRate === null ? '' : `扰动预算使用率为 ${formatRatioPercent(epsilonUsageRate, { clampToUnit: true })}。`
    let comment = ''
    if (epsilonUsageRate !== null && epsilonUsageRate <= 0.3) comment = '当前扰动预算使用较低，说明仍有提升防护强度的空间。'
    else if (epsilonUsageRate !== null && epsilonUsageRate <= 0.8) comment = '当前扰动预算使用适中，防护强度与可听性处于相对平衡状态。'
    else if (epsilonUsageRate !== null) comment = '当前扰动预算使用较高，若听感下降明显，建议降低扰动预算或提高心理声学约束权重。'
    items.push(`指标概览：本次保护扰动 L2 范数为 ${formatMetricValue(l2Norm, 'loss')}，表示保护音频相对原始音频的总体扰动能量。${budget}${comment}`)
  } else {
    missing.push('扰动 L2 范数')
  }

  if (snr !== null) {
    let snrText = 'SNR 处于中等水平，说明扰动已经较明显，但仍可能保持基本听感。'
    if (snr >= 20) snrText = 'SNR 较高，扰动相对较弱，保护音频通常具有较好的可听性。'
    else if (snr < 12) snrText = 'SNR 偏低，说明扰动较强，可能影响正常听感；若目标是高保真保护，可降低 λid / λsem 或提高 λpsy。'
    items.push(`听感质量：${snrText}`)
  } else {
    missing.push('SNR')
  }

  if (pesq !== null || stoi !== null || mos !== null) {
    const qualityNotes: string[] = []
    if (pesq !== null) qualityNotes.push(pesq >= 3 ? 'PESQ 较高，客观感知质量较好。' : pesq >= 2 ? 'PESQ 中等，存在一定感知质量下降。' : 'PESQ 偏低，保护扰动可能明显影响语音质量。')
    if (stoi !== null) qualityNotes.push(stoi >= 0.85 ? 'STOI 较高，语音可懂度保持较好。' : stoi >= 0.65 ? 'STOI 中等，语音可懂度有一定下降。' : 'STOI 偏低，保护音频可能影响正常理解。')
    if (mos !== null) qualityNotes.push(`人工/估计 MOS 为 ${formatMetricValue(mos, 'number')}，可作为主观听感参考。`)
    items.push(`听感质量：${qualityNotes.join('')}`)
  } else {
    items.push('听感质量：PESQ/STOI 未生成，当前感知质量主要依据 SNR、人工 MOS 或音频试听判断。')
  }

  if (overMaskRate !== null || lPsy !== null) {
    const psychoNotes: string[] = []
    if (overMaskRate !== null) {
      psychoNotes.push(overMaskRate <= 0.05 ? '超过听觉掩蔽阈值的频点较少，扰动大多处于较不易察觉区域。' : overMaskRate <= 0.2 ? '部分扰动超过心理声学掩蔽阈值，可能在少量频段产生可察觉噪声。' : '较多扰动超过心理声学掩蔽阈值，高保真性风险较高，建议增大 λpsy 或降低身份/语义攻击权重。')
    }
    if (lPsy !== null) psychoNotes.push(`当前 Lpsy 为 ${formatLossNumber(lPsy)}。Lpsy 表示扰动频谱超过 masking threshold 的惩罚项；数值升高通常意味着扰动更容易被听见。`)
    if (lPsyTrend === 'increasing') psychoNotes.push('Lpsy 在优化过程中呈上升趋势，说明优化正在增强攻击/防护效果的同时牺牲心理声学约束。若希望获得更高保真，可尝试增大 λpsy，或降低 λid / λsem。')
    else if (lPsyTrend === 'decreasing' || lPsyTrend === 'stable') psychoNotes.push('Lpsy 未明显上升，说明心理声学约束相对稳定。')
    items.push(`心理声学：${psychoNotes.join('')}`)
  } else {
    missing.push('心理声学 overMaskRate / Lpsy')
  }

  const lossNotes: string[] = []
  if (lid !== null) lossNotes.push(`Lid 为 ${formatLossNumber(lid)}，表示声音身份特征链路的损失。Lid 的方向需要结合后端 loss 定义解释；当前页面只展示数值趋势，不把单独的 Lid 大小作为最终防护结论。`)
  if (lidTrend === 'increasing') lossNotes.push('Lid 上升可能表示声音身份表示被进一步拉开，通常有利于身份防护，但可能增加可听扰动。')
  if (lidTrend === 'decreasing') lossNotes.push('Lid 下降表示优化器正在降低当前定义下的身份目标损失；若该损失是相似度型 loss，应确认后端是否采用了最小化相似度或最大化距离的等价形式。')
  if (lsem !== null) lossNotes.push(`Lsem 为 ${formatLossNumber(lsem)}，表示语义编码器表示层面的扰动目标。Lsem 变化越明显，通常说明保护音频在语义表示空间中与原始音频差异越大，但具体方向取决于后端 loss 定义。`)
  if (lsem !== null && (Math.abs(lsem) > 1 || lossChanged(trace, 'Lsem'))) lossNotes.push('Lsem 较高或变化明显，说明保护过程对 ASR / tokenizer 语义链路施加了较强影响；若保护音频听感下降，可降低 λsem 或提高 λpsy。')
  items.push(`身份/语义 loss：${lossNotes.length ? lossNotes.join('') : '后端未返回 Lid/Lsem，当前无法解释身份与语义目标的优化状态。'}`)

  const convergenceNotes: string[] = []
  if (l2 !== null) convergenceNotes.push(`L2 为 ${formatLossNumber(l2)}，反映扰动总体能量。L2 持续上升通常意味着保护强度增强，但也可能带来听感下降。`)
  if (total !== null) convergenceNotes.push(`total loss 为 ${formatLossNumber(total)}，是 Lid、Lsem、Lpsy 和 L2 的加权组合，主要用于观察优化过程是否收敛，不应直接等同于最终防护效果。`)
  if (trace.length === 0) convergenceNotes.push('后端未返回逐步 optimizationTrace，因此无法判断各 loss 的收敛趋势，只能展示最终指标。')
  items.push(`优化收敛：${convergenceNotes.length ? convergenceNotes.join('') : '后端返回的 loss 信息不足，暂不判断优化收敛状态。'}`)

  const tuning: string[] = []
  if ((snr !== null && snr < 12) || (pesq !== null && pesq < 2) || (stoi !== null && stoi < 0.65) || (overMaskRate !== null && overMaskRate > 0.2)) {
    tuning.push('建议高保真调参：增大 λpsy，适当降低 λid / λsem，或减少迭代步数与扰动预算。')
  }
  if (epsilonUsageRate !== null && epsilonUsageRate < 0.3 && snr !== null && snr > 20) {
    tuning.push('建议增强防护调参：当前扰动较保守，可适当增大扰动预算、增加迭代步数，或提高 λid / λsem。')
  }
  if (lPsyTrend === 'increasing') tuning.push('建议优先提高 λpsy，使优化器更重视心理声学掩蔽约束。')
  if (l2Trend === 'increasing' && ((pesq !== null && pesq < 2) || (stoi !== null && stoi < 0.65) || (snr !== null && snr < 12))) {
    tuning.push('建议提高 λ2 或降低扰动预算，限制整体扰动能量。')
  }
  if (clippingRate !== null && clippingRate > 0.01) tuning.push('注意：检测到一定比例削波，可降低扰动预算或增加约束以减少失真。')
  items.push(`调参建议：${tuning.length ? tuning.join('') : '当前未触发强风险阈值，可优先结合试听和下游 ASR/克隆评测结果微调 λid、λsem 与 λpsy。'}`)

  if (!psycho?.maskingThreshold?.length || !psycho?.perturbationSpectrum?.length) {
    missing.push('完整心理声学曲线')
  }
  if (trace.length === 0) missing.push('optimizationTrace')
  if (missing.length > 0) items.push(`缺失指标：后端未返回 ${Array.from(new Set(missing)).join('、')}，相关结论会更保守。`)

  while (items.length < 6) items.push('缺失指标：当前保护结果字段仍不完整，建议结合音频试听、ASR 测试和克隆测试共同判断防护效果。')
  return items.slice(0, 10)
}

function generateAsrInsights(asrEval: AsrEval, editStats: EditMetrics | null, result: TaskResult) {
  const wer = optionalNumber(asrEval.wer) ?? (editStats?.level === 'word' ? editStats.werOrCer : null)
  const cer = optionalNumber(asrEval.cer) ?? (editStats?.level === 'char' ? editStats.werOrCer : null)
  const insertRate = optionalNumber(asrEval.insertRate) ?? editStats?.insertRate ?? null
  const tokenChangeRate = optionalNumber(asrEval.tokenChangeRate)
  const semanticDrift = optionalNumber(asrEval.semanticDrift)
  const tokenSource = metricSource(result, ['asrEval.tokenChangeRate'])
  const semanticSource = metricSource(result, ['asrEval.semanticDrift'])?.source
  const semanticSourceKey = String(semanticSource ?? '').toLowerCase()
  const items: string[] = []
  if ((wer ?? 0) >= 0.3 || (cer ?? 0) >= 0.3) items.push('WER/CER 较高，ASR 识别受到干扰。')
  if ((insertRate ?? 0) >= 0.2) items.push('插入率较高，句子结构稳定性下降。')
  if ((tokenChangeRate ?? 0) >= 0.2 && tokenSource?.status === 'available') items.push('语音 tokenizer 序列发生明显变化。')
  if (semanticDrift !== null && semanticSourceKey === 'mfcc_proxy') {
    items.push(`MFCC 代理漂移${semanticDrift >= 0.2 ? '较大' : '较小'}，仅反映声学特征变化。`)
  } else if ((semanticDrift ?? 0) >= 0.2 && semanticSource === 'SemanticEncoderEnsemble') {
    items.push('后端返回的 semanticDrift 较高，语义 encoder 表示发生偏移。')
  }
  if (items.length === 0) items.push('ASR 指标不足或变化较小，当前仅展示后端返回值与文本级 diff，不推断 token 或语义指标。')
  return items
}

function generateCloneInsights(cloneEval: CloneEval) {
  const similarityDropRate = optionalNumber(cloneEval.similarityDropRate) ?? computeDropRate(cloneEval.originalSimilarity, cloneEval.protectedSimilarity)
  const embeddingIncreaseRate = optionalNumber(cloneEval.embeddingDistanceIncreaseRate) ?? computeRateChange(cloneEval.embeddingDistanceBefore, cloneEval.embeddingDistanceAfter)
  const items: string[] = []
  if ((similarityDropRate ?? 0) > 0) items.push('保护后克隆相似度下降，声音身份链路防护有效。')
  if ((embeddingIncreaseRate ?? 0) > 0) items.push('embedding 距离增加，身份表示空间被拉远。')
  if ([cloneEval.originalSimilarity, cloneEval.protectedSimilarity, cloneEval.embeddingDistanceBefore, cloneEval.embeddingDistanceAfter].some((value) => optionalNumber(value) === null)) {
    items.push('部分克隆指标未生成，不做强结论。')
  }
  if (items.length === 0) items.push('语音克隆评估已执行，但后端未返回足够指标用于生成结论。')
  return items
}

function cloneResultToEval(cloneResult?: CloneVoiceResult): CloneEval | null {
  if (!cloneResult) return null
  return {
    cloneModel: cloneResult.request.model,
    targetText: cloneResult.request.text,
    originalCloneAudio: cloneResult.originalCloneAudio,
    protectedCloneAudio: cloneResult.protectedCloneAudio,
  }
}

function TinyWave({ color, className }: { color: string; className?: string }) {
  return (
    <svg viewBox="0 0 520 90" className={cn('h-full w-full', className)} preserveAspectRatio="none">
      <line x1="0" x2="520" y1="45" y2="45" stroke={color} strokeOpacity="0.18" />
      {Array.from({ length: 120 }, (_, index) => {
        const height = 5 + Math.abs(Math.sin(index * 0.52) * 33) + (index % 7) * 2.8
        return <rect key={index} x={index * 4.3} y={(90 - height) / 2} width="1.8" height={height} rx="1" fill={color} opacity={0.33 + (index % 4) * 0.13} />
      })}
    </svg>
  )
}

function LineChart({ result, large }: { result: TaskResult; large?: boolean }) {
  const spectrumPoints =
    result.psychoacoustic?.maskingThreshold && result.psychoacoustic.perturbationSpectrum
      ? result.psychoacoustic.maskingThreshold.map((point) => {
          const matched = result.psychoacoustic?.perturbationSpectrum?.find((item) => item.frequencyHz === point.frequencyHz)
          return {
            frequency: point.frequencyHz,
            maskingThreshold: point.thresholdDb,
            perturbation: matched?.powerDb,
          }
        })
      : []
  const points = spectrumPoints.length ? spectrumPoints : result.charts.psychoacoustic
  const [windowStart, setWindowStart] = useState(0)
  const width = 720
  const height = large ? 220 : 58
  const windowSize = large ? Math.min(points.length, 48) : points.length
  const maxStart = Math.max(0, points.length - windowSize)
  const start = Math.min(windowStart, maxStart)
  const visiblePoints = large && points.length > windowSize ? points.slice(start, start + windowSize) : points
  if (points.length === 0) {
    return <div className="grid h-full place-items-center text-xs text-slate-500">后端未返回心理声学频谱数据</div>
  }
  const values = visiblePoints.flatMap((p) => [p.maskingThreshold, p.perturbation].filter((value): value is number => typeof value === 'number' && Number.isFinite(value)))
  const max = Math.max(...values, 1)
  const min = Math.min(...values, 0)
  const span = Math.max(1, max - min)
  const toPoints = (key: 'maskingThreshold' | 'perturbation') =>
    visiblePoints
      .map((point, index) => {
        const value = point[key]
        if (typeof value !== 'number' || !Number.isFinite(value)) return null
        const x = (index / Math.max(1, visiblePoints.length - 1)) * width
        const y = height - 6 - ((value - min) / span) * (height - 12)
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })
      .filter((point): point is string => point !== null)
      .join(' ')
  const labelEvery = Math.max(1, Math.ceil(visiblePoints.length / 8))
  const firstFrequency = visiblePoints[0]?.frequency
  const lastFrequency = visiblePoints.at(-1)?.frequency

  return (
    <div className="flex h-full min-h-[220px] flex-col">
      <svg viewBox={`0 0 ${width} ${height}`} className={cn('min-h-0 w-full flex-1 overflow-hidden', large ? 'h-full min-h-[190px]' : 'h-[58px]')}>
        {(large ? [34, 78, 122, 166] : [10, 28, 46]).map((y) => (
          <line key={y} x1="0" x2={width} y1={y} y2={y} stroke="rgba(148,163,184,.13)" />
        ))}
        <polyline points={toPoints('maskingThreshold')} fill="none" stroke="#67e8f9" strokeDasharray="6 5" strokeWidth="2" />
        <polyline points={toPoints('perturbation')} fill="none" stroke="#fcd34d" strokeWidth="2" />
        {visiblePoints.filter((_, index) => index % labelEvery === 0).map((point, labelIndex) => (
          <text key={`${point.frequency}-${labelIndex}`} x={labelIndex * labelEvery * (width / Math.max(1, visiblePoints.length - 1))} y={height - 4} fontSize={large ? '11' : '9'} fill="#64748b">
            {point.frequency >= 1000 ? `${Math.round(point.frequency / 1000)}k` : Math.round(point.frequency)}
          </text>
        ))}
      </svg>
      {large && maxStart > 0 ? (
        <div className="mt-2 flex items-center gap-3 text-[10px] text-slate-500">
          <span className="w-14 text-right">{formatFrequency(firstFrequency)}</span>
          <input
            type="range"
            min={0}
            max={maxStart}
            value={start}
            onChange={(event) => setWindowStart(Number(event.target.value))}
            className="h-1 min-w-0 flex-1 accent-cyan-300"
            aria-label="频率范围"
          />
          <span className="w-14">{formatFrequency(lastFrequency)}</span>
        </div>
      ) : null}
    </div>
  )
}

function formatFrequency(value: unknown) {
  const hz = optionalNumber(value)
  if (hz === null) return '-'
  if (hz >= 1000) return `${(hz / 1000).toFixed(hz >= 10000 ? 0 : 1)}k`
  return `${Math.round(hz)}`
}
