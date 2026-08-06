import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  CheckCircle2,
  ChevronDown,
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
import { cloneVoice, downloadEvidenceZip, downloadProtectedAudio, exportReport, getPsychoacousticSlice, getTaskResult, getTaskStatus, listTasks, runAsrEval } from '@/services/apiClient'
import { useCapabilitiesQuery } from '@/hooks/useCapabilitiesQuery'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { AsrEval, AsrMetrics, CapabilitiesResponse, CloneEval, CloneVoiceRequest, CloneVoiceResult, DiffOp, LossFinal, LossTrendPoint, ProtectionRuntimeConfig, PsychoacousticPoint, PsychoacousticSliceResponse, RadarPoint, TaskResult, TaskStatusResponse } from '@/types/task'
import type { AudioFileMeta } from '@/types/audio'
import { downloadBlob } from '@/utils/download'
import { cn } from '@/lib/utils'
import { AudioPlayer } from '@/components/audio/AudioPlayer'
import { formatDurationSeconds, getAudioDuration, getAudioSource } from '@/utils/audio'
import { TrendChart } from '@/components/charts/TrendChart'
import { MathText } from '@/components/common/MathText'
import { cloneMetricDisplay, computeAbsoluteDrop, formatCloneMetricNumber, generateCloneMetricInsights } from '@/utils/cloneMetricDisplay'
import { analyzeLossConvergence, analyzeLossTrend, type TrendDirection } from '@/utils/resultMetrics'

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

const asrWeakDisruptionThreshold = 0.2
const asrStrongDisruptionThreshold = 0.5
const speakerSameIdentityThreshold = 0.25
const speakerHighSimilarityThreshold = 0.5

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
  const [cloneResult, setCloneResult] = useState<CloneVoiceResult | undefined>()
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
  const { data: capabilities } = useCapabilitiesQuery()
  const { data: linkedTaskStatus, refetch: refetchLinkedTaskStatus } = useQuery({
    queryKey: ['task-linked-evaluations', result.taskId],
    queryFn: () => getTaskStatus(result.taskId),
    retry: false,
    refetchInterval: (query) => {
      const status = query.state.data
      const asrStatus = status?.asrTask?.status
      const cloneStatus = status?.cloneTask?.status
      return [asrStatus, cloneStatus].some((value) => value === 'queued' || value === 'running') ? 1500 : false
    },
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
  const linkedAsrResult = linkedTaskStatus?.asrTask?.asrResult ?? linkedTaskStatus?.asrResult
  const linkedCloneResult = linkedTaskStatus?.cloneTask?.cloneResult ?? linkedTaskStatus?.cloneResult
  const activeAsrEval = result.asrEval ?? linkedAsrResult?.asr ?? null
  const originalText = activeAsrEval?.originalText ?? ''
  const referenceText = activeAsrEval?.referenceText ?? originalText
  const protectedText = activeAsrEval?.protectedText ?? ''
  const asrLevel = activeAsrEval?.metricLevel === 'word' || activeAsrEval?.metricLevel === 'char' ? activeAsrEval.metricLevel : chooseEditLevel(referenceText, protectedText)
  const asrEditStats = activeAsrEval && referenceText && protectedText ? computeEditMetrics(referenceText, protectedText, asrLevel) : null
  const activeCloneResult = cloneResult ?? linkedCloneResult ?? result.cloneResults?.at(-1)
  const activeCloneEval = activeCloneResult?.cloneEval ?? cloneResultToEval(activeCloneResult) ?? result.cloneEval ?? null
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
    const originalCloneUrl = activeCloneResult?.originalCloneAudio.objectUrl
    const protectedCloneUrl = activeCloneResult?.protectedCloneAudio.objectUrl
    return () => {
      if (originalUrl?.startsWith('blob:')) URL.revokeObjectURL(originalUrl)
      if (protectedUrl?.startsWith('blob:')) URL.revokeObjectURL(protectedUrl)
      if (originalCloneUrl?.startsWith('blob:')) URL.revokeObjectURL(originalCloneUrl)
      if (protectedCloneUrl?.startsWith('blob:')) URL.revokeObjectURL(protectedCloneUrl)
    }
  }, [activeCloneResult?.originalCloneAudio.objectUrl, activeCloneResult?.protectedCloneAudio.objectUrl, result.originalAudio.objectUrl, result.protectedAudio.objectUrl])

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
      await refetchLinkedTaskStatus()
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
      const asrTask = status.asrTask
      const asrResult = asrTask?.asrResult ?? status.asrResult
      const asrTaskStatus = asrTask?.status ?? (status.stage === 'asr_eval' ? status.status : undefined)
      if (asrResult?.asr) {
        const asrStatus = asrResult.asr.status
        if (asrStatus === 'unavailable' || asrStatus === 'failed' || asrStatus === 'error') {
          throw new Error(asrResult.asr.error || 'ASR 测试失败，请检查后端模型或依赖。')
        }
        return asrResult.asr
      }
      if (asrTaskStatus === 'failed' || asrTaskStatus === 'error') {
        const taskError = asrTask?.error ?? status.error
        throw new Error(typeof taskError === 'string' ? taskError : asrTask?.message || status.message || 'ASR 测试失败，请检查后端服务。')
      }
      if (asrTaskStatus === 'completed' || asrTaskStatus === 'success') {
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
      await refetchLinkedTaskStatus()
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
      const cloneTask = status.cloneTask
      const cloneResult = cloneTask?.cloneResult ?? status.cloneResult
      const cloneTaskState = cloneTask?.status ?? (status.stage === 'downstream_tts_eval' ? status.status : undefined)
      if (cloneTask) {
        setCloneTaskStatus({ ...status, ...cloneTask, taskId: status.taskId } as TaskStatusResponse)
      } else if (status.stage === 'downstream_tts_eval') {
        setCloneTaskStatus(status)
      }
      if (cloneResult) {
        const latest = await getTaskResult(taskId)
        const latestClone = latest.cloneResults?.at(-1)
        if (latestClone) return latestClone
        return cloneResult
      }
      if (cloneTaskState === 'failed' || cloneTaskState === 'error') {
        const taskError = cloneTask?.error ?? status.error
        throw new Error(typeof taskError === 'string' ? taskError : cloneTask?.message || status.message || '语音克隆测试失败，请检查后端服务。')
      }
      if (cloneTaskState === 'completed' || cloneTaskState === 'success') {
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
          <ProtectTab
            result={result}
            originalAudio={originalAudio}
            protectedAudio={protectedAudio}
            linkedTaskStatus={linkedTaskStatus}
            asrEval={activeAsrEval}
            cloneEval={activeCloneEval}
            onProtectedPlayRequest={loadProtectedAudio}
          />
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
  linkedTaskStatus,
  asrEval,
  cloneEval,
  onProtectedPlayRequest,
}: {
  result: TaskResult
  originalAudio: AudioFileMeta
  protectedAudio: AudioFileMeta
  linkedTaskStatus?: TaskStatusResponse
  asrEval?: AsrEval | null
  cloneEval?: CloneEval | null
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
        <div className="compare-badge mx-auto grid h-12 w-12 place-items-center rounded-full border border-cyan-300/28 bg-slate-950/70 text-[18px] font-black text-white shadow-[0_0_24px_rgba(56,189,248,0.12)]">VS</div>
        <AudioCard title="保护音频（已防护）" audio={protectedAudio} color="#22c55e" green onPlayRequest={onProtectedPlayRequest} />
      </div>
      <div className="grid grid-cols-[minmax(360px,0.86fr)_minmax(520px,1.14fr)] items-stretch gap-5 max-xl:grid-cols-1">
        <div className="relative min-h-0 max-xl:static">
          <section className="absolute inset-0 flex min-h-0 flex-col overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4 max-xl:static max-xl:h-auto">
            <SectionTitle>扰动与可听性分析</SectionTitle>
            <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-3">
              <ScoreBox label={<span className="inline-flex items-center justify-center gap-0.5">扰动强度（<MathText formula="L_2" /> 范数）</span>} value={formatMetricValue(perturbation?.l2Norm ?? result.quality.l2Norm, 'loss')} />
              <ScoreBox label="扰动上限利用率" value={formatMetricValue(epsilonUsageRate, 'percent')} />
              <ScoreBox label="信噪比（SNR）" value={formatMetricValue(snr, 'db')} />
            </div>
            <QualityPanel result={result} embedded />
            <MetricGuide />
          </section>
        </div>
        <PsychoacousticPanel result={result} />
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(360px,0.72fr)] items-stretch gap-5 max-xl:grid-cols-1">
        <TrendPanel result={result} embedded />
        <div className="relative min-h-0 max-xl:static">
          <div className="absolute inset-0 min-h-0 max-xl:static">
            <InsightPanel
              title="保护结果解读"
              items={generateProtectionInsights(result, { linkedTaskStatus, asrEval, cloneEval })}
              fillHeight
            />
          </div>
        </div>
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
      <InsightPanel title="ASR 结果解读" items={generateAsrInsights(asrEval, editStats, result)} naturalHeight />
    </div>
  )
}

function CloneTab({ result, cloneEval, loading, status }: { result: TaskResult; cloneEval?: CloneEval | null; loading: boolean; status: TaskStatusResponse | null }) {
  if (loading) {
    return (
      <div className="grid items-center gap-6 pl-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
        <LoadingCard title="克隆原语音" progress={status?.stage === 'downstream_tts_eval' ? status.progress : undefined} message={status?.stage === 'downstream_tts_eval' ? status.message : undefined} />
        <div className="compare-badge mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
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
        <div className="compare-badge mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
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
  const cloneMetrics = cloneMetricDisplay(cloneEval)
  const confidenceBefore = optionalNumber(cloneEval.cloneConfidenceBefore)
  const confidenceAfter = optionalNumber(cloneEval.cloneConfidenceAfter)
  const confidenceDropRate = optionalNumber(cloneEval.cloneConfidenceDropRate)
  const hasCloneConfidence = confidenceBefore !== null || confidenceAfter !== null || confidenceDropRate !== null

  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <SectionTitle>声音身份特征链路分析</SectionTitle>
      <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(210px,1fr))] gap-3">
        <DeltaStatCard
          title="Speaker Similarity（范围 [-1, 1]，越低越好）"
          before={cloneMetrics.similarityBefore}
          after={cloneMetrics.similarityAfter}
          delta={cloneMetrics.similarityDeltaText}
          foot="ECAPA 余弦相似度，范围 [-1, 1]；保护后越低表示越不像原说话人"
          tone="green"
        />
        <DeltaStatCard
          title="Embedding 距离（范围 [0, 2]，越大越好）"
          before={cloneMetrics.embeddingDistanceBefore}
          after={cloneMetrics.embeddingDistanceAfter}
          delta={cloneMetrics.embeddingDistanceDeltaText}
          foot="cosine distance = 1 - similarity，范围 [0, 2]；大于 1 表示相似度已低于 0"
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
    </section>
  )
}

function CloneResultPanel({ cloneEval }: { cloneEval: CloneEval }) {
  const similarityDropAbs = computeAbsoluteDrop(cloneEval.originalSimilarity, cloneEval.protectedSimilarity)

  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <SectionTitle>克隆防护结果</SectionTitle>
      <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3">
        <ScoreBox label="原始克隆相似度" value={formatCloneMetricNumber(cloneEval.originalSimilarity)} />
        <ScoreBox label="保护后克隆相似度" value={formatCloneMetricNumber(cloneEval.protectedSimilarity)} />
        <ScoreBox label="相似度下降量" value={formatCloneMetricNumber(similarityDropAbs)} />
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
    <div className={cn('result-audio-card flex h-[252px] flex-col rounded-[9px] border p-5', green ? 'result-audio-card-protected border-emerald-400/18 bg-emerald-400/8' : 'border-cyan-300/14 bg-[#07192d]/80')}>
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

function ScoreBox({ label, value, red, compact, foot }: { label: ReactNode; value: string; red?: boolean; compact?: boolean; foot?: string }) {
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

const metricGuideGridClass = 'grid grid-cols-[minmax(92px,0.8fr)_64px_minmax(110px,0.9fr)_minmax(0,1.8fr)] gap-x-3 max-md:grid-cols-[minmax(80px,0.8fr)_56px_minmax(92px,0.9fr)_minmax(0,1.6fr)] max-md:gap-x-2'

function MetricGuide() {
  const rows: Array<{ key: string; name: ReactNode; unit: string; range: string; description: string }> = [
    {
      key: 'l2',
      name: <span className="inline-flex items-center gap-1"><MathText formula="L_2" /> 扰动强度</span>,
      unit: '无量纲',
      range: '≥ 0，随音频长度变化',
      description: '整段音频的总体改动量，越小越接近原音。',
    },
    {
      key: 'epsilon',
      name: '扰动上限利用率',
      unit: '%',
      range: '0%～100%',
      description: '已使用的扰动预算比例，越接近 100% 表示越接近设定上限。',
    },
    {
      key: 'snr',
      name: 'SNR',
      unit: 'dB',
      range: '无固定范围，可为负值',
      description: '原始语音与扰动噪声的强弱比，数值越高，音频越接近原音。',
    },
    {
      key: 'pesq',
      name: 'PESQ',
      unit: '分',
      range: '约 -0.5～4.7',
      description: '模拟人耳评价语音质量，分数越高，听感越好。',
    },
    {
      key: 'stoi',
      name: 'STOI',
      unit: '无量纲',
      range: '0～1',
      description: '衡量语音是否容易听懂，越接近 1，可懂度越高。',
    },
    {
      key: 'mos',
      name: 'MOS（人工）',
      unit: '分',
      range: '1～5',
      description: '人工试听给出的主观质量评分，分数越高，听感越好。',
    },
  ]

  return (
    <section className="mt-4 min-h-0 flex-1 overflow-x-hidden overflow-y-auto rounded-[8px] border border-cyan-300/12 bg-slate-950/16 max-xl:max-h-[248px] max-xl:flex-none" aria-labelledby="metric-guide-title">
      <div className="min-w-0 pr-1">
        <div className="px-3 pb-2 pt-3">
          <h3 id="metric-guide-title" className="text-[13px] font-black text-slate-200">指标说明</h3>
          <div className={cn(metricGuideGridClass, 'mt-2 text-[10px] leading-4 text-slate-500')}>
            <span>指标</span>
            <span>单位</span>
            <span>参考范围</span>
            <span>衡量内容</span>
          </div>
        </div>
        <div className="border-t border-cyan-300/10">
          {rows.map((row) => (
            <div key={row.key} className={cn(metricGuideGridClass, 'border-b border-cyan-300/8 px-3 py-2.5 text-[11px] leading-5 last:border-b-0')}>
              <span className="min-w-0 break-words font-bold text-slate-300">{row.name}</span>
              <span className="whitespace-nowrap text-slate-400">{row.unit}</span>
              <span className="min-w-0 break-words text-slate-400">{row.range}</span>
              <span className="min-w-0 break-words text-slate-400">{row.description}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
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
  const pushToast = useAppStore((state) => state.pushToast)
  const [psychoMode, setPsychoMode] = useState<'mean' | 'frame'>('mean')
  const [selectedTimeSec, setSelectedTimeSec] = useState<number | null>(null)
  const [actualTimeSec, setActualTimeSec] = useState<number | null>(null)
  const [frameIndex, setFrameIndex] = useState<number | null>(null)
  const [sliceData, setSliceData] = useState<PsychoacousticSliceResponse | null>(null)
  const [timeDialogOpen, setTimeDialogOpen] = useState(false)
  const [timeDraft, setTimeDraft] = useState('')
  const [sliceLoading, setSliceLoading] = useState(false)
  const [sliceError, setSliceError] = useState<string | null>(null)
  const [modeMenuOpen, setModeMenuOpen] = useState(false)
  const protectedDuration = optionalNumber(result.protectedAudio.durationSec) ?? optionalNumber(result.protectedAudio.duration)
  const originalDuration = optionalNumber(result.originalAudio.durationSec) ?? optionalNumber(result.originalAudio.duration)
  const audioDurationSec = protectedDuration ?? originalDuration ?? getAudioDuration(result.protectedAudio) ?? getAudioDuration(result.originalAudio)
  const chartPoints = psychoMode === 'frame' && sliceData ? psychoPointsFromSlice(sliceData) : psychoPointsFromResult(result)
  const modeLabel = psychoMode === 'frame' ? `t = ${(actualTimeSec ?? selectedTimeSec ?? 0).toFixed(2)} s 对应帧` : 't 平均聚合'
  const modeDescription =
    psychoMode === 'frame'
      ? frameIndex !== null && actualTimeSec !== null
        ? `当前显示 t = ${actualTimeSec.toFixed(2)}s 附近第 ${frameIndex} 帧的心理声学阈值与扰动谱。`
        : '当前显示指定时间附近的单帧心理声学曲线。'
      : '该图为 STFT 时频结果沿时间帧取平均后的频率维曲线。'

  useEffect(() => {
    setPsychoMode('mean')
    setSelectedTimeSec(null)
    setActualTimeSec(null)
    setFrameIndex(null)
    setSliceData(null)
    setTimeDialogOpen(false)
    setTimeDraft('')
    setSliceError(null)
    setModeMenuOpen(false)
  }, [result.taskId])

  const restoreMeanMode = () => {
    setPsychoMode('mean')
    setSelectedTimeSec(null)
    setActualTimeSec(null)
    setFrameIndex(null)
    setSliceData(null)
    setSliceError(null)
    setModeMenuOpen(false)
  }

  const openFrameDialog = () => {
    setModeMenuOpen(false)
    setSliceError(null)
    setTimeDraft(selectedTimeSec !== null ? selectedTimeSec.toFixed(2) : '')
    setTimeDialogOpen(true)
  }

  const confirmFrameTime = async () => {
    const duration = optionalNumber(audioDurationSec)
    if (duration === null) {
      setSliceError('后端未返回音频时长，暂无法指定时间帧。')
      return
    }
    const timeValue = Number(timeDraft)
    if (!Number.isFinite(timeValue) || timeValue < 0 || timeValue > duration) {
      setSliceError(`请输入 0 到 ${duration.toFixed(2)} 秒之间的时间。`)
      return
    }
    setSliceLoading(true)
    setSliceError(null)
    try {
      const response = await getPsychoacousticSlice(result.taskId, { mode: 'frame', timeSec: timeValue })
      setPsychoMode('frame')
      setSelectedTimeSec(timeValue)
      setActualTimeSec(response.actualTimeSec ?? timeValue)
      setFrameIndex(response.frameIndex ?? null)
      setSliceData(response)
      setTimeDialogOpen(false)
    } catch (error) {
      const message = '指定时间帧心理声学曲线加载失败，请稍后重试。'
      setSliceError(message)
      pushToast({ kind: 'error', title: '加载失败', description: error instanceof Error ? error.message : message })
    } finally {
      setSliceLoading(false)
    }
  }

  return (
    <>
      <section className="flex min-h-[296px] flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
        <div className="flex items-center justify-between gap-4 max-md:flex-wrap">
          <SectionTitle>心理声学阈值分析</SectionTitle>
          <div className="flex min-w-[180px] flex-1 justify-center">
            <PsychoacousticModeDropdown label={modeLabel} open={modeMenuOpen} onToggle={() => setModeMenuOpen((open) => !open)} onMean={restoreMeanMode} onFrame={openFrameDialog} />
          </div>
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
          <LineChart result={result} large pointsOverride={chartPoints} />
        </div>
        <p className="mt-3 text-[11px] leading-5 text-slate-500">{modeDescription}</p>
        {sliceError && !timeDialogOpen ? <p className="mt-2 text-[11px] text-rose-300">{sliceError}</p> : null}
      </section>

      {timeDialogOpen ? (
        <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="选择心理声学分析时间点">
          <form
            className="ui-card w-full max-w-[440px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]"
            onSubmit={(event) => {
              event.preventDefault()
              void confirmFrameTime()
            }}
          >
            <div className="mb-5 flex items-center justify-between gap-4">
              <h3 className="text-[20px] font-black text-white">选择心理声学分析时间点</h3>
              <button type="button" onClick={() => setTimeDialogOpen(false)} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="取消">
                <X className="h-4 w-4" />
              </button>
            </div>
            <label className="text-[12px] font-bold text-slate-300" htmlFor="psycho-time-sec">
              时间 t（秒）
            </label>
            <input
              id="psycho-time-sec"
              className="mt-2 h-11 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950/70 px-3 text-slate-100 outline-none focus:border-cyan-300"
              inputMode="decimal"
              onChange={(event) => {
                setTimeDraft(event.target.value)
                setSliceError(null)
              }}
              placeholder="例如 1.25"
              type="number"
              min={0}
              max={optionalNumber(audioDurationSec) ?? undefined}
              step={0.01}
              value={timeDraft}
            />
            <p className="mt-3 text-[12px] leading-5 text-slate-500">
              请输入 0 到 {optionalNumber(audioDurationSec)?.toFixed(2) ?? '未生成'} 秒之间的时间，系统将换算到最接近的 STFT 帧。
            </p>
            {sliceError ? <p className="mt-3 text-[12px] text-rose-300">{sliceError}</p> : null}
            <div className="mt-6 flex justify-end gap-3">
              <button type="button" onClick={() => setTimeDialogOpen(false)} className="h-10 rounded-[7px] border border-cyan-300/14 bg-white/[0.035] px-4 text-sm font-bold text-slate-300 hover:text-white">
                取消
              </button>
              <button type="submit" disabled={sliceLoading} className="cyan-button flex h-10 items-center gap-2 rounded-[7px] px-4 text-sm font-black disabled:cursor-not-allowed disabled:opacity-60">
                {sliceLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                确认查看
              </button>
            </div>
          </form>
        </div>
      ) : null}
    </>
  )
}

function PsychoacousticModeDropdown({ label, open, onToggle, onMean, onFrame }: { label: string; open: boolean; onToggle: () => void; onMean: () => void; onFrame: () => void }) {
  return (
    <div className="relative">
      <button
        type="button"
        onClick={onToggle}
        className="flex h-9 min-w-[168px] items-center justify-center gap-2 rounded-[7px] border border-cyan-300/18 bg-slate-950/55 px-4 text-[12px] font-black text-cyan-100 shadow-[0_0_22px_rgba(34,211,238,0.08)] transition hover:border-cyan-300/36 hover:bg-cyan-300/[0.06]"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span>{label}</span>
        <ChevronDown className={cn('h-3.5 w-3.5 transition', open && 'rotate-180')} />
      </button>
      {open ? (
        <div className="absolute left-1/2 top-11 z-20 w-[180px] -translate-x-1/2 rounded-[8px] border border-cyan-300/18 bg-slate-950/95 p-1 shadow-[0_18px_45px_rgba(0,0,0,0.42)]" role="menu">
          <button type="button" onClick={onMean} className="block h-9 w-full rounded-[6px] px-3 text-left text-[12px] font-bold text-slate-200 hover:bg-cyan-300/[0.08] hover:text-cyan-100" role="menuitem">
            t 平均聚合
          </button>
          <button type="button" onClick={onFrame} className="block h-9 w-full rounded-[6px] px-3 text-left text-[12px] font-bold text-slate-200 hover:bg-cyan-300/[0.08] hover:text-cyan-100" role="menuitem">
            指定 t 对应帧
          </button>
        </div>
      ) : null}
    </div>
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

type LossDisplayKey = 'Lid' | 'Lsem' | 'Lpsy' | 'L2' | 'total'
type LossDefinition = { key: LossDisplayKey; legacyKey?: 'Lfeat'; formula: string; altFormula?: string; label: string; description: string; colorClass: string }

const lossDefinitions: LossDefinition[] = [
  {
    key: 'Lid',
    legacyKey: 'Lfeat',
    formula: 'L_{\\mathrm{id}}',
    label: '声音身份目标',
    description: '声音身份保护目标',
    colorClass: 'bg-cyan-300',
  },
  { key: 'Lsem', formula: 'L_{\\mathrm{sem}}', label: '语义保护目标', description: '语义链路保护目标', colorClass: 'bg-emerald-300' },
  { key: 'Lpsy', formula: 'L_{\\mathrm{psy}}', label: '心理声学目标', description: '心理声学保真目标', colorClass: 'bg-amber-300' },
  { key: 'L2', formula: 'L_2', altFormula: '\\lVert\\delta\\rVert_2', label: '扰动能量约束', description: '扰动范数约束', colorClass: 'bg-violet-300' },
  { key: 'total', formula: 'L_{\\mathrm{total}}', label: '总优化目标', description: '加权总损失', colorClass: 'bg-rose-300' },
]

function TrendPanel({ result, embedded }: { result: TaskResult; embedded?: boolean }) {
  const trend = downsampleTrace(result.optimizationTrace ?? result.generation?.optimizationTrace ?? result.charts.optimizationTrend)
  const lossFinal = result.lossFinal ?? result.generation?.lossFinal ?? finalLossFromTrend(trend)
  const missingLosses = lossDefinitions.filter((loss) => trend.length > 0 && trend.every((point) => lossPointValue(point, loss) === null))
  const totalIterationSteps = lastStep(trend) ?? optionalNumber(result.generation?.steps) ?? optionalNumber(result.generation?.maxSteps)
  const avgIterationSec = optionalNumber(result.averageStepSec) ?? averageStepSecFromTrace(trend) ?? (typeof result.elapsedSec === 'number' && totalIterationSteps && totalIterationSteps > 0 ? result.elapsedSec / totalIterationSteps : null)

  return (
    <section className={cn('flex min-h-0 flex-col overflow-hidden', embedded ? 'rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-5' : 'ui-card p-7')}>
      <SectionTitle>优化损失趋势</SectionTitle>
      <div className="mt-7 grid min-h-0 grid-cols-[minmax(0,1fr)_280px] gap-6 max-lg:grid-cols-1">
        <div className="relative min-h-0 max-lg:static">
          <div className="absolute inset-0 flex min-h-0 flex-col overflow-hidden rounded-[7px] border border-cyan-300/12 bg-slate-950/18 p-5 max-lg:static max-lg:min-h-[560px]">
            <div className="mb-4 flex flex-wrap items-center gap-x-8 gap-y-2 text-[11px] text-slate-400">
              {lossDefinitions.map((loss) => (
                <span key={loss.key} className="inline-flex items-center gap-1.5">
                  <span className={cn('h-2.5 w-2.5 rounded-full', loss.colorClass)} />
                  <MathText formula={loss.formula} />
                  {loss.altFormula ? <MathText formula={loss.altFormula} /> : null}
                </span>
              ))}
            </div>
            {trend.length > 0 ? (
              <div className="min-h-0 flex-1 overflow-hidden">
                <TrendChart data={trend} />
              </div>
            ) : (
              <div className="grid min-h-[210px] flex-1 place-items-center rounded-[6px] border border-dashed border-cyan-300/14 bg-slate-950/16 px-5 text-center text-[12px] leading-5 text-slate-400">
                后端未记录逐步优化损失，当前仅可在详细数据中查看最终 loss。
              </div>
            )}
          </div>
        </div>
        <div className="grid content-start gap-5 pr-1">
          {lossDefinitions.map((loss) => (
            <div key={loss.key} className="rounded-[7px] border border-cyan-300/12 bg-slate-950/18 px-5 py-5">
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
          <div className="rounded-[7px] border border-cyan-300/12 bg-slate-950/20 px-5 py-5">
            <div className="flex items-center justify-between gap-4">
              <p className="text-[11px] font-bold text-slate-400">平均每次迭代耗时</p>
              <p className="text-[14px] font-black text-cyan-200">{avgIterationSec === null ? '未生成' : `${avgIterationSec.toFixed(3)} s / step`}</p>
            </div>
            <div className="mt-3 flex items-center justify-between gap-4 border-t border-cyan-300/10 pt-3">
              <p className="text-[11px] font-bold text-slate-400">总共迭代步数</p>
              <p className="text-[14px] font-black text-white">{totalIterationSteps === null ? '未生成' : `${Math.round(totalIterationSteps)} steps`}</p>
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

function InsightPanel({
  title,
  items,
  fillHeight = false,
  naturalHeight = false,
}: {
  title: string
  items: string[]
  fillHeight?: boolean
  naturalHeight?: boolean
}) {
  return (
    <section className={cn('flex flex-col overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-5', !naturalHeight && 'h-full min-h-[260px]')}>
      <SectionTitle>
        {title} <span className="text-sm font-normal text-slate-500">（自动生成）</span>
      </SectionTitle>
      <div
        className={cn(
          'mt-6 grid grid-cols-1 content-start gap-4 rounded-[7px] border border-cyan-300/10 bg-slate-950/12 p-5 text-[14px] leading-7 text-slate-200',
          !naturalHeight && 'flex-1 overflow-y-auto pr-2',
          !fillHeight && !naturalHeight && 'max-h-[520px]',
        )}
      >
        {items.map((item) => (
          <p key={item} className="flex min-w-0 gap-3">
            <CheckCircle2 className="mt-1.5 h-4 w-4 shrink-0 text-emerald-300" />
            <RichMathText text={item} />
          </p>
        ))}
      </div>
    </section>
  )
}

function RichMathText({ text }: { text: string }) {
  const parts = text.split(/(\\\(.+?\\\))/g).filter(Boolean)
  return (
    <span>
      {parts.map((part, index) => {
        const match = /^\\\((.+)\\\)$/.exec(part)
        return match ? <MathText key={`${part}-${index}`} formula={match[1]} className="mx-0.5 align-[-1px]" /> : <span key={`${part}-${index}`}>{part}</span>
      })}
    </span>
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

function lossTrendText(key: LossDisplayKey, direction: TrendDirection) {
  if (direction === 'insufficient') return '当前优化记录较少，暂不判断该曲线的整体趋势。'
  const messages = {
    Lid: {
      up: '由图所示，\\(L_{\\mathrm{id}}\\) 整体呈上升趋势，当前声音身份保护效果较弱。若更重视声音身份保护，可适当提高 \\(\\lambda_{\\mathrm{id}}\\)。',
      down: '由图所示，\\(L_{\\mathrm{id}}\\) 整体呈下降趋势，声音身份保护目标正在稳定优化。',
      stable: '由图所示，\\(L_{\\mathrm{id}}\\) 整体保持稳定，声音身份保护过程较为平稳。',
    },
    Lsem: {
      up: '由图所示，\\(L_{\\mathrm{sem}}\\) 整体呈上升趋势，当前语义链路保护效果较弱。若更重视语义保护，可适当提高 \\(\\lambda_{\\mathrm{sem}}\\)。',
      down: '由图所示，\\(L_{\\mathrm{sem}}\\) 整体呈下降趋势，语义链路保护目标正在稳定优化。',
      stable: '由图所示，\\(L_{\\mathrm{sem}}\\) 整体保持稳定，语义链路保护过程较为平稳。',
    },
    Lpsy: {
      up: '由图所示，\\(L_{\\mathrm{psy}}\\) 整体呈上升趋势，这一般为正常现象。',
      down: '由图所示，\\(L_{\\mathrm{psy}}\\) 整体呈下降趋势，扰动与心理声学掩蔽范围的匹配正在改善。',
      stable: '由图所示，\\(L_{\\mathrm{psy}}\\) 整体保持稳定，心理声学保真过程较为平稳。',
    },
    L2: {
      up: '由图所示，\\(L_2\\) 整体呈上升趋势，这一般为正常现象。',
      down: '由图所示，\\(L_2\\) 整体呈下降趋势，扰动能量正在得到有效约束。',
      stable: '由图所示，\\(L_2\\) 整体保持稳定，扰动能量控制较为平稳。',
    },
    total: {
      up: '由图所示，\\(L_{\\mathrm{total}}\\) 整体呈上升趋势，优化过程仍在持续变化。',
      down: '由图所示，\\(L_{\\mathrm{total}}\\) 整体呈下降趋势，优化过程正在稳定收敛。',
      stable: '由图所示，\\(L_{\\mathrm{total}}\\) 整体保持稳定，优化过程已进入平稳阶段。',
    },
  } as const
  return messages[key][direction]
}

type ProtectionEvaluationContext = {
  linkedTaskStatus?: TaskStatusResponse
  asrEval?: AsrEval | null
  cloneEval?: CloneEval | null
}

function linkedAsrTuningAdvice({ linkedTaskStatus, asrEval }: ProtectionEvaluationContext) {
  const task = linkedTaskStatus?.asrTask
  const status = task?.status ?? (linkedTaskStatus?.stage === 'asr_eval' ? linkedTaskStatus.status : undefined)
  const linkedResult = task?.asrResult ?? linkedTaskStatus?.asrResult
  const evaluation = linkedResult?.asr ?? asrEval
  const hasTask = Boolean(task || linkedResult || evaluation)

  if (!hasTask) {
    return '调参建议（ASR 联动）：暂未生成对应 ASR 任务，请先点击右上角“ASR 测试”完成识别评估。'
  }
  if (status === 'queued' || status === 'running') {
    return '调参建议（ASR 联动）：对应 ASR 任务正在执行，完成后将根据 WER/CER 自动判断是否需要提高 \\(\\lambda_{\\mathrm{sem}}\\)。'
  }
  if (status === 'failed' || status === 'error') {
    return '调参建议（ASR 联动）：对应 ASR 任务执行失败，请先重新运行 ASR 测试，再根据真实 WER/CER 调整语义权重。'
  }
  if (status === 'cancelled') {
    return '调参建议（ASR 联动）：对应 ASR 任务已取消，请先重新运行 ASR 测试。'
  }
  if (!evaluation || ['unavailable', 'failed', 'error'].includes(String(evaluation.status ?? ''))) {
    return '调参建议（ASR 联动）：对应 ASR 任务暂未返回可用评估结果，请检查 ASR 模型后重试。'
  }

  const level =
    evaluation.metricLevel === 'word' || evaluation.metricLevel === 'char'
      ? evaluation.metricLevel
      : chooseEditLevel(evaluation.referenceText ?? evaluation.originalText ?? '', evaluation.protectedText ?? '')
  const wer = optionalNumber(evaluation.wer)
  const cer = optionalNumber(evaluation.cer)
  const metric = level === 'char' ? cer ?? wer : wer ?? cer
  const metricName = level === 'char' && cer !== null ? 'CER' : wer !== null ? 'WER' : 'CER'
  if (metric === null) {
    return '调参建议（ASR 联动）：对应 ASR 任务已完成，但未返回可用 WER/CER，暂不据此调整 \\(\\lambda_{\\mathrm{sem}}\\)。'
  }

  const metricText = formatRatioPercent(metric)
  if (metric < asrWeakDisruptionThreshold) {
    return `调参建议（ASR 联动）：保护后 ${metricName} 为 ${metricText}，语义干扰较弱；建议提高 \\(\\lambda_{\\mathrm{sem}}\\) 后重新保护并复测。`
  }
  if (metric < asrStrongDisruptionThreshold) {
    return `调参建议（ASR 联动）：保护后 ${metricName} 为 ${metricText}，语义干扰已有一定效果；若优先阻断语义链路，可小幅提高 \\(\\lambda_{\\mathrm{sem}}\\)。`
  }
  return `调参建议（ASR 联动）：保护后 ${metricName} 为 ${metricText}，语义干扰效果较明显，当前可保持 \\(\\lambda_{\\mathrm{sem}}\\)。`
}

function linkedCloneTuningAdvice({ linkedTaskStatus, cloneEval }: ProtectionEvaluationContext) {
  const task = linkedTaskStatus?.cloneTask
  const status = task?.status ?? (linkedTaskStatus?.stage === 'downstream_tts_eval' ? linkedTaskStatus.status : undefined)
  const linkedResult = task?.cloneResult ?? linkedTaskStatus?.cloneResult
  const evaluation = linkedResult?.cloneEval ?? cloneResultToEval(linkedResult ?? undefined) ?? cloneEval
  const hasTask = Boolean(task || linkedResult || evaluation)

  if (!hasTask) {
    return '调参建议（克隆联动）：暂未生成对应克隆任务，请先点击右上角“语音克隆测试”完成声音身份评估。'
  }
  if (status === 'queued' || status === 'running') {
    return '调参建议（克隆联动）：对应克隆任务正在执行，完成后将根据保护后声纹相似度自动判断是否需要提高 \\(\\lambda_{\\mathrm{id}}\\)。'
  }
  if (status === 'failed' || status === 'error') {
    return '调参建议（克隆联动）：对应克隆任务执行失败，请先重新运行语音克隆测试，再根据真实声纹相似度调整身份权重。'
  }
  if (status === 'cancelled') {
    return '调参建议（克隆联动）：对应克隆任务已取消，请先重新运行语音克隆测试。'
  }
  if (!evaluation || ['unavailable', 'failed', 'error'].includes(String(evaluation.status ?? ''))) {
    return '调参建议（克隆联动）：对应克隆任务暂未返回可用声音身份评估，请检查克隆或说话人模型后重试。'
  }

  const originalSimilarity = optionalNumber(evaluation.originalSimilarity)
  const protectedSimilarity = optionalNumber(evaluation.protectedSimilarity)
  const similarityDropRate = optionalNumber(evaluation.similarityDropRate)
  if (protectedSimilarity === null) {
    return '调参建议（克隆联动）：对应克隆任务已完成，但未返回保护后声纹相似度，暂不据此调整 \\(\\lambda_{\\mathrm{id}}\\)。'
  }

  const similarityText = protectedSimilarity.toFixed(3)
  const dropText = similarityDropRate === null ? '' : `，较原始克隆下降 ${formatRatioPercent(similarityDropRate)}`
  if (originalSimilarity !== null && originalSimilarity < speakerSameIdentityThreshold) {
    return `调参建议（克隆联动）：原始音频克隆的声纹相似度为 ${originalSimilarity.toFixed(3)}，本次克隆基线偏弱；建议先更换克隆模型或样本复测，暂不据此调整 \\(\\lambda_{\\mathrm{id}}\\)。`
  }
  if (protectedSimilarity >= speakerHighSimilarityThreshold) {
    return `调参建议（克隆联动）：保护后克隆声纹相似度为 ${similarityText}${dropText}，声音身份残留较高；建议提高 \\(\\lambda_{\\mathrm{id}}\\) 后重新保护并复测。`
  }
  if (protectedSimilarity >= speakerSameIdentityThreshold) {
    return `调参建议（克隆联动）：保护后克隆声纹相似度为 ${similarityText}${dropText}，仍有一定声音身份特征残留；建议小幅提高 \\(\\lambda_{\\mathrm{id}}\\)。`
  }
  return `调参建议（克隆联动）：保护后克隆声纹相似度为 ${similarityText}${dropText}，声音身份相似度已明显降低，当前可保持 \\(\\lambda_{\\mathrm{id}}\\)。`
}

function generateProtectionInsights(result: TaskResult, evaluationContext: ProtectionEvaluationContext = {}) {
  const perturbation = result.perturbation
  const quality = result.protectionQuality ?? result.quality
  const trace = downsampleTrace(result.optimizationTrace ?? result.generation?.optimizationTrace ?? result.charts?.optimizationTrend ?? [])
  const snr = optionalNumber(perturbation?.snr) ?? optionalNumber(quality?.snr)
  const pesq = optionalNumber(quality?.pesq)
  const stoi = optionalNumber('stoi' in (quality ?? {}) ? (quality as { stoi?: number | null }).stoi : null)
  const epsilonUsageRate = optionalNumber(perturbation?.epsilonUsageRate) ?? computeEpsilonUsageRate(perturbation)
  const trends = {
    Lid: analyzeLossTrend(trace, 'Lid'),
    Lsem: analyzeLossTrend(trace, 'Lsem'),
    Lpsy: analyzeLossTrend(trace, 'Lpsy'),
    L2: analyzeLossTrend(trace, 'L2'),
    total: analyzeLossTrend(trace, 'total'),
  }
  const convergence = analyzeLossConvergence(trace)
  const items: string[] = []

  if (epsilonUsageRate === null) {
    items.push('指标概览：扰动预算使用率尚未完成评估。')
  } else {
    const usage = formatRatioPercent(epsilonUsageRate, { clampToUnit: true })
    const strength = epsilonUsageRate < 0.7 ? '保护强度较为保守' : epsilonUsageRate < 0.9 ? '保护强度处于中等水平' : '当前保护强度较高'
    items.push(`指标概览：本次保护的扰动预算使用率为 ${usage}，${strength}。`)
  }

  const qualityNotes: string[] = []
  if (snr !== null) {
    const level = snr >= 25 ? '整体听感质量良好' : snr >= 18 ? '整体听感质量中等' : '当前噪声较明显，听感质量较弱'
    qualityNotes.push(`SNR 为 ${snr.toFixed(2)} dB，${level}。`)
  }
  if (pesq !== null) {
    const level = pesq >= 3 ? '语音感知质量良好' : pesq >= 2 ? '语音感知质量中等' : '语音感知质量较弱'
    qualityNotes.push(`PESQ 为 ${pesq.toFixed(2)}，${level}。`)
  }
  if (stoi !== null) {
    const level = stoi >= 0.9 ? '语音可懂度良好' : stoi >= 0.75 ? '语音可懂度中等' : '语音可懂度较弱'
    qualityNotes.push(`STOI 为 ${stoi.toFixed(3)}，${level}。`)
  }
  items.push(`听感质量：${qualityNotes.length ? qualityNotes.join('') : '该指标尚未完成评估。'}`)
  items.push(`心理声学保真：${lossTrendText('Lpsy', trends.Lpsy.direction)}`)
  items.push(`身份保护：${lossTrendText('Lid', trends.Lid.direction)}`)
  items.push(`语义保护：${lossTrendText('Lsem', trends.Lsem.direction)}`)
  items.push(`扰动能量：${lossTrendText('L2', trends.L2.direction)}`)
  items.push(`优化收敛：${lossTrendText('total', trends.total.direction)}`)

  const tuning: string[] = []
  if (trends.Lid.direction === 'up' && trends.Lsem.direction === 'up') {
    tuning.push('由图所示，身份保护与语义保护曲线整体呈上升趋势。若需要强化对应保护，可适当提高 \\(\\lambda_{\\mathrm{id}}\\) 与 \\(\\lambda_{\\mathrm{sem}}\\)。')
  } else {
    if (trends.Lid.direction === 'up') tuning.push('可适当提高 \\(\\lambda_{\\mathrm{id}}\\) 强化声音身份保护。')
    if (trends.Lsem.direction === 'up') tuning.push('可适当提高 \\(\\lambda_{\\mathrm{sem}}\\) 强化语义保护。')
  }
  if (trends.Lpsy.direction === 'up') tuning.push('可适当提高 \\(\\lambda_{\\mathrm{psy}}\\) 强化听感保真。')
  if (trends.L2.direction === 'up') tuning.push('可适当提高 \\(\\lambda_2\\) 约束扰动能量。')
  if (convergence.status === 'unconverged') tuning.push('如图所示，可能没有收敛，为了更好的效果可以尝试增大迭代次数。')
  if (snr !== null && snr < 18) tuning.push('若优先提升听感，可适当降低扰动预算。')
  items.push(`调参建议：${tuning.length ? tuning.join('') : '当前曲线运行平稳，保持现有参数即可。'}`)
  items.push(linkedAsrTuningAdvice(evaluationContext))
  items.push(linkedCloneTuningAdvice(evaluationContext))
  return items
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
    items.push(`语义漂移为 ${semanticDrift?.toFixed(3)}，语义 encoder 表示发生偏移。`)
  }
  if (items.length === 0) items.push('ASR 指标不足或变化较小，当前仅展示后端返回值与文本级 diff，不推断 token 或语义指标。')
  return items
}

function generateCloneInsights(cloneEval: CloneEval) {
  return generateCloneMetricInsights(cloneEval)
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

function psychoPointsFromResult(result: TaskResult): PsychoacousticPoint[] {
  if (result.psychoacoustic?.maskingThreshold?.length && result.psychoacoustic.perturbationSpectrum?.length) {
    const perturbationByFrequency = new Map(result.psychoacoustic.perturbationSpectrum.map((item) => [item.frequencyHz, item.powerDb]))
    return result.psychoacoustic.maskingThreshold.map((point) => ({
      frequency: point.frequencyHz,
      maskingThreshold: point.thresholdDb,
      perturbation: perturbationByFrequency.get(point.frequencyHz),
    }))
  }
  return result.charts.psychoacoustic
}

function psychoPointsFromSlice(slice: PsychoacousticSliceResponse): PsychoacousticPoint[] {
  if (slice.charts?.psychoacoustic?.length) return slice.charts.psychoacoustic
  const perturbationByFrequency = new Map(slice.perturbationSpectrum.map((item) => [item.frequencyHz, item.powerDb]))
  return slice.maskingThreshold.map((point) => ({
    frequency: point.frequencyHz,
    maskingThreshold: point.thresholdDb,
    perturbation: perturbationByFrequency.get(point.frequencyHz),
  }))
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

function LineChart({ result, large, pointsOverride }: { result: TaskResult; large?: boolean; pointsOverride?: PsychoacousticPoint[] }) {
  const points = pointsOverride ?? psychoPointsFromResult(result)
  const [windowStart, setWindowStart] = useState(0)
  const width = 720
  const height = large ? 220 : 58
  const windowSize = large ? Math.min(points.length, 48) : points.length
  const maxStart = Math.max(0, points.length - windowSize)
  const start = Math.min(windowStart, maxStart)
  const visiblePoints = large && points.length > windowSize ? points.slice(start, start + windowSize) : points
  useEffect(() => {
    setWindowStart(0)
  }, [points.length])
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
