import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ClipboardList,
  Clock3,
  Copy,
  Download,
  Fingerprint,
  Info,
  Loader2,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  TestTube2,
  Volume2,
  Waves,
  X,
} from 'lucide-react'
import { cloneVoice, downloadProtectedAudio, getPsychoacousticSlice, getTaskResult, getTaskStatus, listTasks, runAsrEval } from '@/services/apiClient'
import { useCapabilitiesQuery } from '@/hooks/useCapabilitiesQuery'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { AsrEval, AsrEvalResponse, AsrMetrics, CapabilitiesResponse, CloneEval, CloneVoiceRequest, CloneVoiceResult, DiffOp, EvaluationBatch, LossFinal, LossTrendPoint, ProtectionEvaluation, ProtectionEvaluationDimension, ProtectionEvaluationDimensionKey, ProtectionRuntimeConfig, PsychoacousticPoint, PsychoacousticSliceResponse, RuntimeModelOption, SubtaskStatusSnapshot, TaskResult, TaskStatusResponse } from '@/types/task'
import type { AudioFileMeta } from '@/types/audio'
import { downloadBlob } from '@/utils/download'
import { cn } from '@/lib/utils'
import { AudioPlayer } from '@/components/audio/AudioPlayer'
import { getAudioDuration, getAudioSource } from '@/utils/audio'
import { TrendChart } from '@/components/charts/TrendChart'
import { MathBlock, MathText } from '@/components/common/MathText'
import { ModelInformationModal } from '@/components/common/ModelInformationModal'
import { computeAbsoluteDelta, formatCloneMetricNumber, generateCloneMetricInsights } from '@/utils/cloneMetricDisplay'
import { formatAsrRatePercent, generateAsrMetricInsights } from '@/utils/asrInsightDisplay'
import { analyzeLossConvergence, analyzeLossTrend, type TrendDirection } from '@/utils/resultMetrics'
import { resolveEpsilonUsageRate } from '@/utils/perturbationMetrics'
import { resolveAsrErrorShares } from '@/utils/metricNormalization'
import { seconds } from '@/utils/format'
import { cloneModelRequiresReferenceText, normalizeCloneReferenceTextRequest } from '@/utils/cloneModelCapabilities'
import { defaultCloneTextForLanguage, translateDefaultCloneText } from '@/utils/cloneDefaultText'
import { lossTrendSeries } from '@/utils/lossTrendSeries'

const statusText: Record<TaskResult['status'], string> = {
  queued: '排队中',
  running: '处理中',
  completed: '已完成',
  success: '已完成',
  partial_failed: '部分失败',
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
      modelTypes: capabilities.modelTypes,
      constraints: capabilities.constraints,
    }
  }
  return undefined
}

type BackendSelectOption = RuntimeModelOption & { label: string; value: string }

function backendOptionItems(options?: ProtectionRuntimeConfig['models'][string]) {
  return (options ?? [])
    .map((option) =>
      typeof option === 'string'
        ? { label: option, name: option, value: option }
        : {
            ...option,
            label: option.label ?? option.value,
            value: option.value,
          },
    )
    .filter((option) => option.value)
}

function isAvailableModel(option: BackendSelectOption) {
  return option.status === undefined || option.status === 'available'
}

function normalizeEvaluationLanguage(value?: string | null) {
  return String(value ?? 'zh-cn').toLowerCase().startsWith('zh') ? 'zh-cn' : 'en'
}

function preferredAsrModel(options: BackendSelectOption[], language: string) {
  const available = options.filter(isAvailableModel)
  const chinese = normalizeEvaluationLanguage(language) === 'zh-cn'
  const preferred = chinese
    ? available.find((option) => option.type?.includes('chinese_asr') || option.value === 'funasr:paraformer-zh')
    : available.find((option) => option.value === 'openai-whisper:medium')
  return preferred?.value ?? available.find((option) => option.value.includes('whisper'))?.value ?? available[0]?.value ?? ''
}

const asrWeakDisruptionThreshold = 0.2
const asrStrongDisruptionThreshold = 0.5
const speakerSameIdentityThreshold = 0.25
const speakerHighSimilarityThreshold = 0.5

const delay = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

function hasRunningLinkedEvaluation(status?: TaskStatusResponse) {
  const subtaskStates = [
    ...(status?.asrTasks ?? []).map((task) => task.status),
    ...(status?.cloneTasks ?? []).map((task) => task.status),
    status?.asrTask?.status,
    status?.cloneTask?.status,
    ...(status?.asrBatches ?? []).map((batch) => batch.status),
    ...(status?.cloneBatches ?? []).map((batch) => batch.status),
  ]
  return subtaskStates.some((value) => value === 'queued' || value === 'running')
}

function computeCloneDefenseScore(cloneEval?: CloneEval | null) {
  if (!cloneEval) return null
  return optionalNumber(cloneEval.cloneIdentityScore) ?? optionalNumber(cloneEval.cloneDefenseScore)
}

function summarizeCloneDefenseScore(result: TaskResult, status?: TaskStatusResponse) {
  const aggregate = result.protectionEvaluation?.dimensions.find((dimension) => dimension.key === 'cloneIdentity')
  const aggregateScore = optionalNumber(aggregate?.score)
  if (aggregateScore !== null) {
    const count = (result.cloneResults ?? []).filter((item) => computeCloneDefenseScore(item.cloneEval ?? cloneResultToEval(item)) !== null).length
    return { count: Math.max(1, count), score: aggregateScore }
  }
  const candidates = [
    ...(result.cloneResults ?? []),
    ...(status?.cloneTasks ?? []).map((task) => task.cloneResult),
    status?.cloneTask?.cloneResult,
    status?.cloneResult,
  ].filter((item): item is CloneVoiceResult => Boolean(item?.cloneId))
  const unique = new Map<string, CloneVoiceResult>()
  candidates.forEach((item) => unique.set(item.cloneId, item))
  const scores = Array.from(unique.values())
    .map((item) => computeCloneDefenseScore(item.cloneEval ?? cloneResultToEval(item)))
    .filter((score): score is number => score !== null)
  if (!scores.length) {
    const fallbackScore = computeCloneDefenseScore(result.cloneEval)
    if (fallbackScore !== null) scores.push(fallbackScore)
  }
  return {
    count: scores.length,
    score: scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : null,
  }
}

export function ResultsPage() {
  const { taskId } = useParams()
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)
  const setCurrentTaskResult = useTaskStore((state) => state.setCurrentTaskResult)
  const missingTaskGuardRef = useRef(false)
  const [asrOverrideState, setAsrOverrideState] = useState<{ taskId: string; asr: AsrMetrics }>()
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
          description: '请先选择一条有效的保护记录。',
          dedupeMs: 5000,
        })
        navigate('/workspace', { replace: true })
      })
      .catch(() => {
        pushToast({
          id: 'results-missing-task-id',
          kind: 'error',
          title: '请先进行音频保护任务',
          description: '请先选择一条有效的保护记录。',
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
          <p className="mt-4 text-slate-300">正在加载保护结果...</p>
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
      <SummaryBar
        result={displayData}
        onTaskInfoClick={() => {
          const displayFilename = resultDisplayFilename(displayData)
          navigator.clipboard.writeText(displayFilename)
          pushToast({ kind: 'success', title: '已复制', description: `音频文件名 ${displayFilename} 已复制到剪贴板。` })
        }}
        onDownloadClick={() => setDownloadOpen(true)}
      />

      <AudioCompare result={displayData} onAsrUpdated={(asr) => setAsrOverrideState({ taskId, asr })} />
      {downloadOpen ? <DownloadModal result={displayData} onClose={() => setDownloadOpen(false)} /> : null}
    </div>
  )
}

function SummaryBar({ result, onTaskInfoClick, onDownloadClick }: { result: TaskResult; onTaskInfoClick: () => void; onDownloadClick: () => void }) {
  const { data: linkedTaskStatus } = useQuery({
    queryKey: ['task-linked-evaluations', result.taskId],
    queryFn: () => getTaskStatus(result.taskId),
    retry: false,
    refetchInterval: (query) => (hasRunningLinkedEvaluation(query.state.data) ? 1500 : false),
  })
  const cloneScore = summarizeCloneDefenseScore(result, linkedTaskStatus)
  const overallScore = optionalNumber(result.protectionEvaluation?.overallScore)
  const taskScore = optionalNumber(result.score)
  const displayedScore = overallScore ?? taskScore ?? cloneScore.score
  const scoreSource = overallScore !== null ? 'overall' : taskScore !== null ? 'task' : cloneScore.score !== null ? 'clone' : 'none'
  const hasScore = displayedScore !== null
  const scoreText = displayedScore === null ? '未生成' : `${displayedScore.toFixed(2)} 分`
  const scoreTitle = scoreSource === 'overall'
    ? '综合防护评分'
    : scoreSource === 'task'
      ? '保护任务评分'
      : scoreSource === 'clone'
        ? `基于 ${cloneScore.count} 次可用克隆评估的身份保护结果`
        : '等待语音克隆测试'

  return (
    <section className="ui-card grid min-h-[74px] grid-cols-[250px_180px_250px_170px_230px_minmax(290px,1fr)] items-center px-5 max-2xl:grid-cols-[1.05fr_0.78fr_1.08fr_0.75fr_1fr_1.55fr] max-xl:h-auto max-xl:grid-cols-3 max-xl:gap-y-4 max-xl:py-4">
      <SummaryItem icon={<ClipboardList />} label="保护任务" value={resultDisplayFilename(result)} copy buttonTitle="点击复制音频文件名" onClick={onTaskInfoClick} />
      <SummaryItem icon={<ShieldCheck />} label="保护状态" value={statusText[result.status] ?? result.status} green={result.status === 'completed' || result.status === 'success'} />
      <SummaryItem icon={<Clock3 />} label="完成时间" value={result.completedAt ?? '-'} />
      <SummaryItem icon={<Clock3 />} label="处理耗时" value={typeof result.elapsedSec === 'number' ? formatElapsed(result.elapsedSec) : '-'} />
      <SummaryItem icon={<Sparkles />} label="防护模式" value={modeText[result.mode] ?? result.mode} green />
      <button
        type="button"
        onClick={onDownloadClick}
        title={scoreTitle}
        className="flex h-full min-h-[58px] items-center justify-center gap-3 border-l border-cyan-300/10 pl-5 transition hover:bg-cyan-400/[0.035]"
      >
        {hasScore ? <ShieldCheck className="h-11 w-11 text-cyan-300" /> : null}
        <p className={cn('shrink-0 font-mono text-[27px] font-black leading-none', hasScore ? 'text-emerald-300' : 'text-rose-300')}>
          {scoreText}
        </p>
        <div className="min-w-0 text-left">
          <p className="truncate text-[16px] font-black leading-tight text-cyan-100">{hasScore ? '保护结果已生成' : '等待克隆测试'}</p>
          <p className="mt-1 truncate text-xs text-slate-400">{hasScore ? '点击此处下载' : '完成评估后自动更新'}</p>
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

function MetricInfoDialog({
  open,
  title,
  children,
  onClose,
  wide = false,
}: {
  open: boolean
  title: ReactNode
  children: ReactNode
  onClose: () => void
  wide?: boolean
}) {
  useEffect(() => {
    if (!open) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose, open])

  if (!open) return null
  return createPortal(
    <div className="fixed inset-0 z-[240] grid place-items-center bg-slate-950/80 px-4 py-8" role="dialog" aria-modal="true" aria-label="指标说明" onClick={onClose}>
      <div className={cn('ui-card max-h-full w-full overflow-y-auto !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.56)]', wide ? 'max-w-[1012px]' : 'max-w-[572px]')} onClick={(event) => event.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-bold tracking-[0.12em] text-cyan-300">指标说明</p>
            <h3 className="mt-2 text-lg font-black text-white">{title}</h3>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-cyan-300/14 text-slate-300 hover:text-white" aria-label="关闭指标说明">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-5 rounded-[8px] border border-cyan-300/12 bg-slate-950 p-4 text-sm leading-7 text-slate-300">
          {typeof children === 'string' ? <RichMathText text={children} /> : children}
        </div>
      </div>
    </div>,
    document.body,
  )
}

function MetricInfoButton({ title, children, wide = false }: { title: ReactNode; children: ReactNode; wide?: boolean }) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <button
        type="button"
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
          setOpen(true)
        }}
        className="grid h-7 w-7 shrink-0 place-items-center rounded-[6px] border border-cyan-300/16 text-cyan-200 transition hover:border-cyan-300/32 hover:bg-cyan-300/10"
        aria-label="查看指标说明"
        title="查看说明"
      >
        <Search className="h-3.5 w-3.5" />
      </button>
      <MetricInfoDialog open={open} title={title} wide={wide} onClose={() => setOpen(false)}>{children}</MetricInfoDialog>
    </>
  )
}

function MetricInfoSurface({
  title,
  info,
  children,
  className,
  tooltip,
}: {
  title: ReactNode
  info?: ReactNode
  children: ReactNode
  className?: string
  tooltip?: string
}) {
  const [open, setOpen] = useState(false)
  const interactive = Boolean(info)
  const accessibleTitle = typeof title === 'string' ? title : '该指标'

  return (
    <>
      <div
        className={cn(
          className,
          interactive && 'cursor-pointer transition duration-200 ease-out hover:-translate-y-0.5 hover:border-cyan-300/32 hover:bg-cyan-300/[0.035] hover:shadow-[0_14px_38px_rgba(34,211,238,0.10)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/45 motion-reduce:transform-none motion-reduce:transition-none',
        )}
        role={interactive ? 'button' : undefined}
        tabIndex={interactive ? 0 : undefined}
        aria-label={interactive ? `查看${accessibleTitle}说明` : undefined}
        title={tooltip ?? (interactive ? '点击查看指标说明' : undefined)}
        onClick={interactive ? () => setOpen(true) : undefined}
        onKeyDown={interactive ? (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            setOpen(true)
          }
        } : undefined}
      >
        {children}
      </div>
      {info ? <MetricInfoDialog open={open} title={title} onClose={() => setOpen(false)}>{info}</MetricInfoDialog> : null}
    </>
  )
}

type ComparePanel = 'protect' | 'clone'
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

type AsrHistoryEntry = AsrEvalResponse & {
  taskStatus?: SubtaskStatusSnapshot
}

type CloneHistoryEntry = {
  key: string
  taskId: string
  cloneSubId?: string
  cloneId?: string
  status: string
  request?: CloneVoiceRequest
  result?: CloneVoiceResult
  taskStatus?: SubtaskStatusSnapshot
  createdAt?: string | null
}

function AudioCompare({ result, onAsrUpdated }: { result: TaskResult; onAsrUpdated: (asr: AsrMetrics) => void }) {
  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const pushToast = useAppStore((state) => state.pushToast)
  const queryClient = useQueryClient()
  const initialEvaluationLanguage = normalizeEvaluationLanguage(result.asr.language ?? result.cloneResults?.at(-1)?.request?.language)
  const [activePanel, setActivePanel] = useState<ComparePanel>('protect')
  const [protectedObjectUrl, setProtectedObjectUrl] = useState<string>()
  const [cloneModalOpen, setCloneModalOpen] = useState(false)
  const [cloneLoading, setCloneLoading] = useState(false)
  const [cloneError, setCloneError] = useState<string>()
  const [cloneResult, setCloneResult] = useState<CloneVoiceResult | undefined>()
  const [selectedCloneKey, setSelectedCloneKey] = useState<string>()
  const [selectedAsrSubId, setSelectedAsrSubId] = useState<string>()
  const [cloneTaskStatus, setCloneTaskStatus] = useState<TaskStatusResponse | null>(null)
  const [asrModalOpen, setAsrModalOpen] = useState(false)
  const [asrLoading, setAsrLoading] = useState(false)
  const [asrError, setAsrError] = useState<string>()
  const [asrModel, setAsrModel] = useState('')
  const [asrLanguage, setAsrLanguage] = useState(initialEvaluationLanguage)
  const cloneTextUsesDefaultRef = useRef(!result.asr.originalText)
  const [cloneForm, setCloneForm] = useState<CloneVoiceRequest>({
    text: result.asr.originalText || defaultCloneTextForLanguage(initialEvaluationLanguage),
    model: result.cloneResults?.at(-1)?.request?.model ?? '',
    language: initialEvaluationLanguage,
    speed: 1,
    speakerPrompt: result.asr.originalText || '',
  })
  const { data: capabilities } = useCapabilitiesQuery()
  const { data: linkedTaskStatus, refetch: refetchLinkedTaskStatus } = useQuery({
    queryKey: ['task-linked-evaluations', result.taskId],
    queryFn: () => getTaskStatus(result.taskId),
    retry: false,
    refetchInterval: (query) => (hasRunningLinkedEvaluation(query.state.data) ? 1500 : false),
  })
  const runtimeConfig = configFromCapabilities(capabilities)
  const protectionModelName = result.generation?.source?.trim() || result.processingModel?.trim() || 'VoiceShield.protect'
  const configuredAsrOptions = useMemo(() => backendOptionItems(runtimeConfig?.models.asr), [runtimeConfig?.models.asr])
  const asrOptions = useMemo(
    () => {
      if (configuredAsrOptions.length) return configuredAsrOptions
      const value = result.asrModel || result.asr.model
      return value ? [{ label: value, name: value, value, status: 'available' as const }] : []
    },
    [configuredAsrOptions, result.asr.model, result.asrModel],
  )
  const asrValues = useMemo(() => asrOptions.filter(isAvailableModel).map((option) => option.value), [asrOptions])
  const effectiveAsrModel = asrValues.includes(asrModel) ? asrModel : preferredAsrModel(asrOptions, asrLanguage)
  const configuredTtsOptions = useMemo(
    () => backendOptionItems(runtimeConfig?.models.tts),
    [runtimeConfig?.models.tts],
  )
  // New clone requests must only use the backend-owned capability catalog.
  // Historical clone models are rendered from their saved request/result data
  // and must never be promoted back into an available creation option while
  // capabilities are loading or unavailable.
  const ttsModelOptions = configuredTtsOptions
  const ttsOptions = useMemo(() => ttsModelOptions.filter(isAvailableModel).map((option) => option.value), [ttsModelOptions])
  const selectedTtsOption = ttsModelOptions.find((option) => option.value === cloneForm.model) ?? ttsModelOptions[0]
  const oneClickAsrLimit = Math.max(1, asrOptions.filter(isAvailableModel).length)
  const oneClickCloneLimit = Math.max(1, ttsModelOptions.filter(isAvailableModel).reduce((count, option) => count + (cloneModelRequiresReferenceText(option) ? 2 : 1), 0))
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
  const asrHistory = useMemo<AsrHistoryEntry[]>(() => {
    const snapshots = [
      ...(linkedTaskStatus?.asrTasks ?? []),
      ...(linkedTaskStatus?.asrTask ? [linkedTaskStatus.asrTask] : []),
    ]
    const snapshotById = new Map<string, SubtaskStatusSnapshot>()
    snapshots.forEach((snapshot) => {
      if (snapshot.asrSubId) snapshotById.set(snapshot.asrSubId, snapshot)
    })

    const results = [
      ...(result.asrResults ?? []),
      ...snapshots.map((snapshot) => snapshot.asrResult).filter((item): item is AsrEvalResponse => Boolean(item)),
      linkedTaskStatus?.asrResult,
    ].filter((item): item is AsrEvalResponse => Boolean(item))
    const unique = new Map<string, AsrHistoryEntry>()
    results.forEach((item, index) => {
      const key = item.asrSubId || `legacy-${item.createdAt || index}`
      unique.set(key, { ...item, taskStatus: item.asrSubId ? snapshotById.get(item.asrSubId) : undefined })
    })
    snapshots.forEach((snapshot, index) => {
      const asrSubId = snapshot.asrSubId ?? undefined
      const key = asrSubId || `snapshot-${snapshot.createdAt || index}`
      const current = unique.get(key)
      const resultEntry = snapshot.asrResult?.asr ? snapshot.asrResult : current ?? snapshot.asrResult
      unique.set(key, {
        taskId: resultEntry?.taskId ?? result.taskId,
        status: resultEntry?.status ?? String(snapshot.status ?? 'queued'),
        asr: resultEntry?.asr,
        asrSubId,
        request: resultEntry?.request ?? snapshot.asrRequest ?? undefined,
        createdAt: resultEntry?.createdAt ?? snapshot.createdAt,
        taskStatus: snapshot,
      })
    })
    return Array.from(unique.values()).sort((left, right) => String(left.taskStatus?.createdAt ?? left.createdAt ?? '').localeCompare(String(right.taskStatus?.createdAt ?? right.createdAt ?? '')))
  }, [linkedTaskStatus, result.asrResults, result.taskId])
  const cloneHistory = useMemo<CloneHistoryEntry[]>(() => {
    const persistedCloneResults = result.cloneResults ?? []
    const snapshots = [
      ...(linkedTaskStatus?.cloneTasks ?? []),
      ...(linkedTaskStatus?.cloneTask ? [linkedTaskStatus.cloneTask] : []),
    ]
    const snapshotBySubId = new Map<string, SubtaskStatusSnapshot>()
    snapshots.forEach((snapshot) => {
      if (snapshot.cloneSubId) snapshotBySubId.set(snapshot.cloneSubId, snapshot)
    })
    const candidates = [
      ...(result.cloneResults ?? []),
      ...snapshots.map((task) => task.cloneResult).filter((item): item is CloneVoiceResult => Boolean(item)),
      linkedTaskStatus?.cloneResult,
      cloneResult,
    ].filter((item): item is CloneVoiceResult => Boolean(item?.cloneId || item?.cloneSubId))
    const unique = new Map<string, CloneHistoryEntry>()
    candidates.forEach((item, index) => {
      const key = item.cloneSubId ? `sub:${item.cloneSubId}` : item.cloneId ? `clone:${item.cloneId}` : `legacy:${index}`
      unique.set(key, {
        key,
        taskId: item.taskId || result.taskId,
        cloneSubId: item.cloneSubId,
        cloneId: item.cloneId || undefined,
        status: item.status,
        request: item.request,
        result: item,
        taskStatus: item.cloneSubId ? snapshotBySubId.get(item.cloneSubId) : undefined,
        createdAt: item.cloneEval?.createdAt,
      })
    })
    snapshots.forEach((snapshot, index) => {
      const cloneSubId = snapshot.cloneSubId ?? undefined
      const key = cloneSubId ? `sub:${cloneSubId}` : `snapshot:${snapshot.createdAt ?? index}`
      const current = unique.get(key)
      const persistedResult = cloneSubId
        ? persistedCloneResults.find((item) => item.cloneSubId === cloneSubId)
        : snapshot.cloneResult?.cloneId
          ? persistedCloneResults.find((item) => item.cloneId === snapshot.cloneResult?.cloneId)
          : undefined
      const snapshotResult = persistedResult ?? snapshot.cloneResult ?? current?.result
      unique.set(key, {
        key,
        taskId: snapshotResult?.taskId ?? current?.taskId ?? result.taskId,
        cloneSubId,
        cloneId: snapshotResult?.cloneId || current?.cloneId,
        status: String(snapshot.status ?? snapshotResult?.status ?? current?.status ?? 'queued'),
        request: snapshot.cloneRequest ?? snapshotResult?.request ?? current?.request,
        result: snapshotResult ?? current?.result,
        taskStatus: snapshot,
        createdAt: snapshot.createdAt ?? snapshotResult?.cloneEval?.createdAt ?? current?.createdAt,
      })
    })
    return Array.from(unique.values()).sort((left, right) => String(left.createdAt ?? '').localeCompare(String(right.createdAt ?? '')))
  }, [cloneResult, linkedTaskStatus, result.cloneResults, result.taskId])
  const selectedAsrResult = selectedAsrSubId ? asrHistory.find((item) => item.asrSubId === selectedAsrSubId) : undefined
  const latestCompletedAsrResult = [...asrHistory].reverse().find((item) => item.asr)
  const activeAsrResult = selectedAsrSubId ? selectedAsrResult : latestCompletedAsrResult ?? asrHistory.at(-1)
  const activeAsrEval = selectedAsrSubId ? selectedAsrResult?.asr ?? null : activeAsrResult?.asr ?? result.asrEval ?? null
  const originalText = activeAsrEval?.originalText ?? ''
  const referenceText = activeAsrEval?.referenceText ?? originalText
  const protectedText = activeAsrEval?.protectedText ?? ''
  const asrLevel = activeAsrEval?.metricLevel === 'word' || activeAsrEval?.metricLevel === 'char' ? activeAsrEval.metricLevel : chooseEditLevel(referenceText, protectedText)
  const asrEditStats = activeAsrEval && referenceText && protectedText ? computeEditMetrics(referenceText, protectedText, asrLevel) : null
  const selectedCloneEntry = selectedCloneKey ? cloneHistory.find((item) => item.key === selectedCloneKey) : undefined
  const activeCloneEntry = selectedCloneKey ? selectedCloneEntry : cloneHistory.at(-1)
  const activeCloneResult = activeCloneEntry?.result
  // Clone detail cards intentionally read one active history result. The
  // aggregated protectionEvaluation dimensions never participate here.
  const activeCloneEval = activeCloneResult?.cloneEval
    ?? cloneResultToEval(activeCloneResult)
    ?? (activeCloneEntry ? null : result.cloneEval ?? null)
  const activeCloneKey = selectedCloneKey ?? activeCloneEntry?.key
  const completedCloneHistory = cloneHistory.map((item) => item.result).filter((item): item is CloneVoiceResult => Boolean(item))
  const compareTabs = [
    {
      key: 'protect',
      label: '语音保护结果',
    },
    {
      key: 'clone',
      label: '克隆测试结果',
    },
  ] as const

  const scrollToResult = (id: string) => {
    window.requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  }

  const openAsrHistoryResult = (asrSubId?: string) => {
    if (asrSubId) setSelectedAsrSubId(asrSubId)
    setActivePanel('clone')
    scrollToResult('asr-result-detail')
  }

  const openCloneHistoryResult = (cloneKey: string) => {
    setSelectedCloneKey(cloneKey)
    setActivePanel('clone')
    scrollToResult('clone-result-detail')
  }

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
      const nextModel = defaultCloneConfig?.backendValue || defaultCloneConfig?.model || ttsOptions[0] || ''
      const nextModelOption = ttsModelOptions.find((option) => option.value === nextModel || option.backendValue === nextModel) ?? ttsModelOptions.find((option) => ttsOptions.includes(option.value)) ?? ttsModelOptions[0]
      const preferredLanguage = defaultCloneConfig?.uiPreferredLanguage || defaultCloneConfig?.language || 'zh-cn'
      const nextModelLanguages = nextModelOption?.languages?.length ? nextModelOption.languages : cloneLanguages
      const nextLanguage = nextModelLanguages.includes(preferredLanguage) ? preferredLanguage : nextModelLanguages[0] || 'en'
      const nextSpeed = defaultCloneConfig?.speed ?? cloneSpeeds[0] ?? 1
      setCloneForm((current) => {
        const currentModelOption = ttsModelOptions.find((option) => option.value === current.model)
        const currentLanguages = currentModelOption?.languages?.length ? currentModelOption.languages : cloneLanguages
        const currentLanguageSupported = currentLanguages.includes(current.language ?? '')
        const model = currentModelOption && currentLanguageSupported && ttsOptions.includes(current.model) ? current.model : nextModelOption?.value ?? ttsOptions[0]
        const modelOption = ttsModelOptions.find((option) => option.value === model) ?? nextModelOption
        const modelLanguages = modelOption?.languages?.length ? modelOption.languages : cloneLanguages
        const language = modelLanguages.includes(current.language ?? '') ? current.language : nextLanguage
        return {
          ...current,
          model,
          language,
          text: cloneTextUsesDefaultRef.current && normalizeEvaluationLanguage(language) !== normalizeEvaluationLanguage(current.language)
            ? translateDefaultCloneText(current.text, language)
            : current.text,
          speed: cloneSpeeds.includes(Number(current.speed)) ? current.speed : nextSpeed,
        }
      })
    }, 0)
    return () => window.clearTimeout(timeoutId)
  }, [cloneLanguages, cloneSpeeds, defaultCloneConfig, runtimeConfig, ttsModelOptions, ttsOptions])

  const changeAsrLanguage = (language: string) => {
    setAsrLanguage(language)
    setAsrModel(preferredAsrModel(asrOptions, language))
  }

  const changeCloneForm = (nextForm: CloneVoiceRequest) => {
    const language = normalizeEvaluationLanguage(nextForm.language)
    const languageChanged = language !== normalizeEvaluationLanguage(cloneForm.language)
    const textChanged = nextForm.text !== cloneForm.text
    if (textChanged) cloneTextUsesDefaultRef.current = false
    if (languageChanged) {
      setAsrLanguage(language)
      setAsrModel(preferredAsrModel(asrOptions, language))
    }
    setCloneForm(languageChanged && !textChanged && cloneTextUsesDefaultRef.current
      ? { ...nextForm, text: translateDefaultCloneText(nextForm.text, language) }
      : nextForm)
  }

  const submitAsrTest = async (modelOverride = effectiveAsrModel) => {
    if (!asrValues.includes(modelOverride)) {
      setAsrError('请选择当前可用的 ASR 模型。')
      setAsrModalOpen(true)
      return
    }
    try {
      setAsrLoading(true)
      setAsrError(undefined)
      setAsrModalOpen(false)
      setActivePanel('clone')
      const response = await runAsrEval(result.taskId, { model: modelOverride, language: asrLanguage, referenceText: referenceText || originalText || result.asr.referenceText || result.asr.originalText || undefined })
      if (response.asrSubId) setSelectedAsrSubId(response.asrSubId)
      await refetchLinkedTaskStatus()
      const asr = response.asr ?? (await waitForAsrEvalResult(result.taskId, response.asrSubId))
      await refetchLinkedTaskStatus()
      onAsrUpdated(asr)
      await queryClient.invalidateQueries({ queryKey: ['task-result', result.taskId] })
      pushToast({ kind: 'success', title: 'ASR 测试完成', description: asr.model ?? modelOverride })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'ASR 测试失败，请检查服务状态。'
      setAsrError(message)
      setAsrModalOpen(true)
      pushToast({ kind: 'error', title: 'ASR 测试失败', description: message })
    } finally {
      setAsrLoading(false)
    }
  }

  const submitQuickAsrTest = async () => {
    const model = preferredAsrModel(asrOptions, asrLanguage)
    if (!model) {
      setAsrError(`${asrLanguage === 'zh-cn' ? '中文' : '英文'}推荐 ASR 模型当前不可用。`)
      return
    }
    setAsrModel(model)
    await submitAsrTest(model)
  }

  const waitForAsrEvalResult = async (taskId: string, asrSubId?: string) => {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const status = await getTaskStatus(taskId)
      const asrTask = asrSubId ? status.asrTasks?.find((task) => task.asrSubId === asrSubId) : status.asrTask
      const asrResult = asrTask?.asrResult ?? status.asrResult
      const asrTaskStatus = asrTask?.status ?? (status.stage === 'asr_eval' ? status.status : undefined)
      if (asrResult?.asr) {
        const asrStatus = asrResult.asr.status
        if (asrStatus === 'unavailable' || asrStatus === 'failed' || asrStatus === 'error') {
          throw new Error(asrResult.asr.error || 'ASR 测试失败，请稍后重试或更换模型。')
        }
        return asrResult.asr
      }
      if (asrTaskStatus === 'failed' || asrTaskStatus === 'error') {
        const taskError = asrTask?.error ?? status.error
        throw new Error(typeof taskError === 'string' ? taskError : asrTask?.message || status.message || 'ASR 测试失败，请检查服务状态。')
      }
      if (asrTaskStatus === 'completed' || asrTaskStatus === 'success') {
        const latest = await getTaskResult(taskId)
        if (asrSubId) {
          const matchingAsr = latest.asrResults?.find((item) => item.asrSubId === asrSubId)?.asr
          if (matchingAsr) {
            if (matchingAsr.status === 'unavailable' || matchingAsr.status === 'failed' || matchingAsr.status === 'error') {
              throw new Error(matchingAsr.error || 'ASR 测试失败，请稍后重试或更换模型。')
            }
            return matchingAsr
          }
          await delay(250)
          continue
        }
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

  const validateCloneRequest = (request: CloneVoiceRequest) => {
    const modelOption = cloneModelOption(ttsModelOptions, request.model)
    const modelLanguages = modelOption?.languages?.length ? modelOption.languages : cloneLanguages
    if (!request.text.trim()) return '请输入用于语音克隆的文本。'
    if (!modelOption || !isAvailableModel(modelOption)) return '请选择当前可用的克隆模型。'
    if (cloneModelRequiresReferenceText(modelOption) && request.annotationSource !== 'asr' && !request.speakerPrompt?.trim()) return '所选模型需要参考音频对应文本，请填写人工标注。'
    if (cloneModelRequiresReferenceText(modelOption) && request.annotationSource === 'asr' && (!request.annotationAsrSubId || !request.originalSpeakerPrompt?.trim() || !request.protectedSpeakerPrompt?.trim())) return '所选模型需要参考音频对应文本，请选择一条同时包含原始音频和保护音频转写的 ASR 标注。'
    if (!modelLanguages.includes(request.language ?? '')) return '请选择当前模型支持的克隆语言。'
    if (!cloneSpeeds.includes(Number(request.speed))) return '请选择当前支持的克隆语速。'
    return undefined
  }

  const submitCloneTest = async (requestOverride = cloneForm) => {
    const modelOption = cloneModelOption(ttsModelOptions, requestOverride.model)
    const request = normalizeCloneReferenceTextRequest(requestOverride, modelOption)
    const validationError = validateCloneRequest(request)
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
      const response = await cloneVoice(result.taskId, { ...request, text: request.text.trim() })
      if (response.cloneSubId) setSelectedCloneKey(`sub:${response.cloneSubId}`)
      await refetchLinkedTaskStatus()
      const nextResult =
        (response.status === 'completed' || response.status === 'success') && getAudioSource(response.originalCloneAudio) && getAudioSource(response.protectedCloneAudio)
          ? response
          : await waitForCloneResult(result.taskId, response.cloneSubId)
      setCloneResult(nextResult)
      setSelectedCloneKey(nextResult.cloneSubId ? `sub:${nextResult.cloneSubId}` : `clone:${nextResult.cloneId}`)
      await refetchLinkedTaskStatus()
      await queryClient.invalidateQueries({ queryKey: ['task-result', result.taskId] })
      pushToast({ kind: 'success', title: '语音克隆测试完成', description: nextResult.message ?? '克隆结果和评分已更新。' })
    } catch (error) {
      const message = error instanceof Error ? error.message : '语音克隆测试失败，请检查表单或服务状态。'
      setCloneError(message)
      setCloneModalOpen(true)
      pushToast({ kind: 'error', title: '语音克隆测试失败', description: message })
    } finally {
      setCloneLoading(false)
    }
  }

  const submitQuickCloneTest = async () => {
    const requestedLanguage = normalizeEvaluationLanguage(cloneForm.language)
    const currentOption = ttsModelOptions.find((option) => option.value === cloneForm.model && isAvailableModel(option) && (option.languages?.includes(requestedLanguage) ?? true))
    const modelOption = currentOption ?? ttsModelOptions.find((option) => isAvailableModel(option) && (option.languages?.includes(requestedLanguage) ?? true))
    if (!modelOption) {
      setCloneError(`当前没有支持${requestedLanguage === 'zh-cn' ? '中文' : '英文'}的一键克隆模型。`)
      return
    }

    const latestAnnotation = [...asrHistory].reverse().find((item) => item.asrSubId && item.asr?.originalText?.trim() && item.asr?.protectedText?.trim())
    const manualPrompt = cloneForm.speakerPrompt?.trim() || referenceText || originalText || result.asr.originalText || ''
    const usesDefaultFallback = !cloneForm.text.trim()
    const quickRequest: CloneVoiceRequest = {
      ...cloneForm,
      text: usesDefaultFallback ? defaultCloneTextForLanguage(requestedLanguage) : cloneForm.text.trim(),
      model: modelOption.value,
      language: requestedLanguage,
      speed: cloneSpeeds.includes(Number(cloneForm.speed)) ? cloneForm.speed : 1,
    }
    if (cloneModelRequiresReferenceText(modelOption) && latestAnnotation) {
      quickRequest.annotationSource = 'asr'
      quickRequest.annotationAsrSubId = latestAnnotation.asrSubId
      quickRequest.annotationAsrModel = latestAnnotation.asr?.model
      quickRequest.annotationCreatedAt = latestAnnotation.createdAt ?? undefined
      quickRequest.speakerPrompt = latestAnnotation.asr?.originalText ?? ''
      quickRequest.originalSpeakerPrompt = latestAnnotation.asr?.originalText ?? ''
      quickRequest.protectedSpeakerPrompt = latestAnnotation.asr?.protectedText ?? ''
    } else if (cloneModelRequiresReferenceText(modelOption)) {
      quickRequest.annotationSource = 'manual'
      quickRequest.speakerPrompt = manualPrompt
      quickRequest.originalSpeakerPrompt = undefined
      quickRequest.protectedSpeakerPrompt = undefined
      quickRequest.annotationAsrSubId = undefined
      quickRequest.annotationAsrModel = undefined
      quickRequest.annotationCreatedAt = undefined
    }
    const normalizedQuickRequest = normalizeCloneReferenceTextRequest(quickRequest, modelOption)
    if (usesDefaultFallback) cloneTextUsesDefaultRef.current = true
    setCloneForm(normalizedQuickRequest)
    await submitCloneTest(normalizedQuickRequest)
  }

  const waitForCloneResult = async (taskId: string, cloneSubId?: string) => {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const status = await getTaskStatus(taskId)
      const cloneTask = cloneSubId ? status.cloneTasks?.find((task) => task.cloneSubId === cloneSubId) : status.cloneTask
      const cloneResult = cloneTask?.cloneResult ?? status.cloneResult
      const cloneTaskState = cloneTask?.status ?? (status.stage === 'downstream_tts_eval' ? status.status : undefined)
      if (cloneTask) {
        setCloneTaskStatus({ ...status, ...cloneTask, taskId: status.taskId } as TaskStatusResponse)
      } else if (status.stage === 'downstream_tts_eval') {
        setCloneTaskStatus(status)
      }
      if (cloneResult) {
        const latest = await getTaskResult(taskId)
        const latestClone = cloneSubId ? latest.cloneResults?.find((item) => item.cloneSubId === cloneSubId) : latest.cloneResults?.at(-1)
        if (latestClone) return latestClone
        return cloneResult
      }
      if (cloneTaskState === 'failed' || cloneTaskState === 'error') {
        const taskError = cloneTask?.error ?? status.error
        throw new Error(typeof taskError === 'string' ? taskError : cloneTask?.message || status.message || '语音克隆测试失败，请检查服务状态。')
      }
      if (cloneTaskState === 'completed' || cloneTaskState === 'success') {
        const latest = await getTaskResult(taskId)
        const latestClone = cloneSubId ? latest.cloneResults?.find((item) => item.cloneSubId === cloneSubId) : latest.cloneResults?.at(-1)
        if (latestClone) return latestClone
      }
      await delay(1000)
    }
    throw new Error('语音克隆测试仍在执行，请稍后刷新结果页查看。')
  }

  return (
    <section className="ui-card h-full p-5">
      <div className="relative flex min-h-9 items-center">
        <div className="flex flex-wrap items-center gap-3">
          <SectionTitle info>{protectionModelName} 的保护结果对比</SectionTitle>
          <div className="flex items-center gap-2">
            {compareTabs.map(({ key, label }) => (
              <button key={key} type="button" onClick={() => setActivePanel(key as ComparePanel)} className={cn('h-9 rounded-[7px] border border-cyan-300/14 px-3 text-sm font-black text-slate-300 transition hover:text-white', activePanel === key && 'bg-cyan-400/14 text-cyan-200')} title={`查看${label}结果`}>
                {label}
              </button>
            ))}
          </div>
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
            asrHistory={asrHistory}
            cloneHistory={completedCloneHistory}
            onProtectedPlayRequest={loadProtectedAudio}
          />
        ) : null}
        {activePanel === 'clone' ? (
          <CloneTab result={result} cloneResult={activeCloneResult} cloneEval={activeCloneEval} cloneHistory={cloneHistory} cloneBatches={linkedTaskStatus?.cloneBatches ?? []} selectedCloneKey={activeCloneKey} onSelectClone={openCloneHistoryResult} onOpenAsr={openAsrHistoryResult} loading={cloneLoading} status={cloneTaskStatus} cloneModelOptions={ttsModelOptions} modelTypes={runtimeConfig?.modelTypes} asrEval={activeAsrEval} asrEditStats={asrEditStats} asrHistory={asrHistory} asrBatches={linkedTaskStatus?.asrBatches ?? []} selectedAsrSubId={selectedAsrSubId ?? activeAsrResult?.asrSubId} asrHistoryLimit={oneClickAsrLimit} cloneHistoryLimit={oneClickCloneLimit} />
        ) : null}
      </div>
      {cloneModalOpen ? (
        <CloneVoiceModal
          form={cloneForm}
          error={cloneError}
          loading={cloneLoading}
          modelOptions={ttsModelOptions}
          modelTypes={runtimeConfig?.modelTypes}
          languageOptions={cloneLanguages}
          speedOptions={cloneSpeeds}
          asrAnnotations={asrHistory}
          onChange={changeCloneForm}
          onClose={() => setCloneModalOpen(false)}
          onSubmit={() => void submitCloneTest()}
          onQuickSubmit={() => void submitQuickCloneTest()}
          onOpenAsr={() => {
            setCloneModalOpen(false)
            setActivePanel('clone')
            setAsrModalOpen(true)
          }}
        />
      ) : null}
      {asrModalOpen ? (
        <AsrEvalModal
          model={effectiveAsrModel}
          error={asrError}
          loading={asrLoading}
          modelOptions={asrOptions}
          modelTypes={runtimeConfig?.modelTypes}
          language={asrLanguage}
          onLanguageChange={changeAsrLanguage}
          onChange={setAsrModel}
          onClose={() => setAsrModalOpen(false)}
          onSubmit={() => void submitAsrTest()}
          onQuickSubmit={() => void submitQuickAsrTest()}
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
  asrHistory,
  cloneHistory,
  onProtectedPlayRequest,
}: {
  result: TaskResult
  originalAudio: AudioFileMeta
  protectedAudio: AudioFileMeta
  linkedTaskStatus?: TaskStatusResponse
  asrEval?: AsrEval | null
  cloneEval?: CloneEval | null
  asrHistory?: AsrEvalResponse[]
  cloneHistory?: CloneVoiceResult[]
  onProtectedPlayRequest: () => Promise<string | undefined>
}) {
  const perturbation = result.perturbation
  const epsilonUsageRate = resolveEpsilonUsageRate(perturbation)
  const directDistance = optionalNumber(result.speaker.directDistance)
    ?? optionalNumber(result.speaker.embeddingDistance)
    ?? optionalNumber(result.speaker.embeddingDistanceAfter)
  const hasEvaluationExperiment = Boolean(
    asrEval
      || cloneEval
      || asrHistory?.length
      || cloneHistory?.length
      || linkedTaskStatus?.asrResult
      || linkedTaskStatus?.cloneResult
      || linkedTaskStatus?.asrTask
      || linkedTaskStatus?.cloneTask
      || linkedTaskStatus?.asrTasks?.length
      || linkedTaskStatus?.cloneTasks?.length
      || linkedTaskStatus?.asrBatches?.length
      || linkedTaskStatus?.cloneBatches?.length,
  )

  return (
    <div className="space-y-5">
      <div className="grid items-center gap-6 pr-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
        <AudioCard title="原始音频（未保护）" audio={originalAudio} color="#00aef0" compactHeader />
        <div className="compare-badge mx-auto grid h-12 w-12 place-items-center rounded-full border border-cyan-300/28 bg-slate-950/70 text-[18px] font-black text-white shadow-[0_0_24px_rgba(56,189,248,0.12)]">VS</div>
        <AudioCard title="保护音频（已防护）" audio={protectedAudio} color="#22c55e" green compactHeader onPlayRequest={onProtectedPlayRequest} />
      </div>
      <div className="grid grid-cols-[minmax(360px,0.86fr)_minmax(520px,1.14fr)] items-stretch gap-5 max-xl:grid-cols-1">
        <div className="min-h-0">
          <section className="flex min-h-0 flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
            <SectionTitle>扰动与直接保护效果</SectionTitle>
            <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-3">
              <ScoreBox
                label={<span className="inline-flex items-center justify-center gap-0.5">扰动强度</span>}
                value={formatMetricValue(perturbation?.l2Norm ?? result.quality.l2Norm, 'loss')}
                foot={(
                  <MetricFormulaContent
                    description="先计算保护音频与原始音频的逐点差值，再统计整段音频的总体改动量。"
                    formulas={[
                      '\\delta=x^{\\prime}-x',
                      '\\lVert\\delta\\rVert_2=\\sqrt{\\sum_{n=1}^{N}\\delta_n^2}',
                    ]}
                    note="数值越小，表示保护音频整体越接近原始音频。"
                  />
                )}
              />
              <ScoreBox
                label="扰动上限利用率"
                value={formatRatioPercent(epsilonUsageRate)}
                foot={(
                  <MetricFormulaContent
                    description={<>在 <MathText formula={'L_{\\infty}'} className="mx-0.5 align-[-1px]" /> 模式下，使用最大单点扰动与设定上限的比值。</>}
                    formulas={["U_{\\epsilon}=\\frac{\\lVert x^{\\prime}-x\\rVert_{\\infty}}{\\epsilon}\\times 100\\%"]}
                    note={<>即采样点扰动幅度与设定上限的最大比值。</>}
                  />
                )}
              />
              <ScoreBox
                label="直接声纹偏移"
                value={formatMetricValue(directDistance, 'number')}
                foot={(
                  <MetricFormulaContent
                    description="独立声纹模型先计算原音频与保护音频的声音身份相似度，再转换为距离。"
                    formulas={[
                      "\\operatorname{SIM}(x,x^{\\prime})=\\cos(\\operatorname{Emb}(x),\\operatorname{Emb}(x^{\\prime}))",
                      "D_{\\mathrm{direct}}=1-\\operatorname{SIM}(x,x^{\\prime})",
                    ]}
                    note="距离越大，表示两段音频的声音身份越分离。"
                  />
                )}
              />
            </div>
            <QualityPanel result={result} embedded />
          </section>
        </div>
        <div className="relative min-h-0 max-xl:min-h-[296px]">
          <div className="absolute inset-0 max-xl:static max-xl:h-[296px]">
            <PsychoacousticPanel key={result.taskId} result={result} />
          </div>
        </div>
      </div>
      <div className="grid grid-cols-[minmax(0,1.15fr)_minmax(420px,1fr)] items-stretch gap-5 max-xl:grid-cols-1">
        <TrendPanel result={result} embedded />
        <div className="relative min-h-0 max-xl:static">
          <div className="absolute inset-0 min-h-0 max-xl:static">
            <InsightPanel
              title="保护结果解读"
              items={generateProtectionInsights(result, { linkedTaskStatus, asrEval, cloneEval, asrHistory, cloneHistory })}
              fillHeight
            />
          </div>
        </div>
      </div>
      {hasEvaluationExperiment ? <ComprehensiveProtectionEvaluation evaluation={result.protectionEvaluation} /> : null}
    </div>
  )
}

const protectionDimensionOrder: ProtectionEvaluationDimensionKey[] = [
  'protectionQuality',
  'cloneQuality',
  'protectionSemantic',
  'cloneSemantic',
  'directIdentity',
  'cloneIdentity',
]

const protectionDimensionLabels: Record<ProtectionEvaluationDimensionKey, string> = {
  protectionQuality: '保护音频听感质量',
  cloneQuality: '克隆音频质量下降',
  protectionSemantic: '保护后音频语义干扰',
  cloneSemantic: '克隆后音频语义干扰',
  directIdentity: '保护后声音身份直接保护效果',
  cloneIdentity: '克隆声音身份保护效果',
}

const protectionRadarLabels: Record<ProtectionEvaluationDimensionKey, string> = {
  protectionQuality: '保护听感质量',
  cloneQuality: '克隆质量下降',
  protectionSemantic: '保护语义干扰',
  cloneSemantic: '克隆语义干扰',
  directIdentity: '直接身份保护',
  cloneIdentity: '克隆身份保护',
}

function resultDisplayFilename(result: TaskResult) {
  return result.originalAudio.displayFilename || result.originalAudio.filename || result.protectedAudio.displayFilename || result.protectedAudio.filename || '音频文件名待生成'
}

const protectionDimensionInfo: Record<ProtectionEvaluationDimensionKey, ReactNode> = {
  protectionQuality: (
    <MetricFormulaContent
      description="综合信噪比、语音感知质量、可懂度和语音质量评分，判断保护音频是否仍然清楚、自然、易于理解。"
      formulas={['S_{\\mathrm{quality}}=0.40S_{\\mathrm{SNR}}+0.35S_{\\mathrm{STOI}}+0.15S_{\\mathrm{PESQ}}+0.10S_{\\mathrm{DNSMOS}}']}
      note="分数越高，表示保护音频在听感和可懂度方面保持得越好。"
    />
  ),
  cloneQuality: (
    <MetricFormulaContent
      description="将原音频生成的克隆音频作为参考，与保护音频生成的克隆音频比较 PESQ、STOI 和 DNSMOS，衡量克隆后的听感下降。"
      formulas={[
        "Q=0.45S_{\\mathrm{PESQ}}+0.45S_{\\mathrm{STOI}}+0.10S_{\\mathrm{DNSMOS}}",
        "Q^{\\prime}=0.45S_{\\mathrm{PESQ}}^{\\prime}+0.45S_{\\mathrm{STOI}}^{\\prime}+0.10S_{\\mathrm{DNSMOS}}^{\\prime}",
        "d_q=\\max\\!\\left(0,\\frac{Q-Q^{\\prime}}{Q}\\right)",
        'S_{q}^{\\mathrm{raw}}=\\Phi(d_q;d_{90})',
      ]}
      note="分数越高，表示保护使克隆音频的质量下降越明显；最终评分也会结合身份和语义保护效果判断这项下降的参考价值。"
    />
  ),
  protectionSemantic: (
    <MetricFormulaContent
      description="综合离散语音 Token 的变化比例和语义表示的漂移程度，衡量保护音频对内容识别的干扰。"
      formulas={['S_{\\mathrm{sem}}=0.55S_{\\mathrm{token}}+0.45S_{\\mathrm{drift}}']}
      note="分数越高，表示保护音频越难被稳定还原为原有表达内容。"
    />
  ),
  cloneSemantic: (
    <MetricFormulaContent
      description="比较原始克隆语音和保护后克隆语音的离散 Token 与语义表示，衡量克隆后表达内容受到的干扰。"
      formulas={["S_{\\mathrm{clone\\_sem}}=0.55S_{\\mathrm{token}}^{\\prime}+0.45S_{\\mathrm{drift}}^{\\prime}"]}
      note="分数越高，表示克隆语音越难保持原来的表达内容。"
    />
  ),
  directIdentity: (
    <MetricFormulaContent
      description="比较原始音频和保护音频的声纹相似度，再将声音身份距离转换为分数。"
      formulas={["D_{\\mathrm{id}}=1-\\operatorname{SIM}(x,x^{\\prime})", 'S_{\\mathrm{direct\\_id}}=\\Phi(D_{\\mathrm{id}};D_{90})']}
      note="分数越高，表示保护音频与原说话人的声音身份差异越明显。"
    />
  ),
  cloneIdentity: (
    <MetricFormulaContent
      description="比较保护前后两段克隆语音与原说话人的声纹距离，衡量保护使克隆声音远离原身份的程度。"
      formulas={["d=1-\\operatorname{SIM}(x,c),\\qquad d^{\\prime}=1-\\operatorname{SIM}(x,c^{\\prime})", "S_{\\mathrm{clone\\_id}}=95P(d,d^{\\prime})+B(d^{\\prime})"]}
      note="分数越高，表示保护后生成的克隆声音越难保持原说话人的声音身份。"
    />
  ),
}

const protectionDimensionWeights: Record<ProtectionEvaluationDimensionKey, number> = {
  protectionQuality: 0.2,
  cloneQuality: 0.1,
  protectionSemantic: 0.2,
  cloneSemantic: 0.15,
  directIdentity: 0.15,
  cloneIdentity: 0.2,
}

function protectionDimensionIcon(key: ProtectionEvaluationDimensionKey) {
  if (key === 'protectionQuality') return <Volume2 className="h-4 w-4" />
  if (key === 'cloneQuality') return <Waves className="h-4 w-4" />
  if (key === 'protectionSemantic') return <BrainCircuit className="h-4 w-4" />
  if (key === 'cloneSemantic') return <Sparkles className="h-4 w-4" />
  if (key === 'directIdentity') return <Fingerprint className="h-4 w-4" />
  return <ShieldCheck className="h-4 w-4" />
}

function completeProtectionDimensions(evaluation?: ProtectionEvaluation | null) {
  const byKey = new Map((evaluation?.dimensions ?? []).map((dimension) => [dimension.key, dimension]))
  return protectionDimensionOrder.map((key): ProtectionEvaluationDimension => byKey.get(key) ?? {
    key,
    label: protectionDimensionLabels[key],
    score: null,
    status: 'pending',
    reason: '待生成',
    weight: protectionDimensionWeights[key],
  })
}

function useAnimatedScore(value: number | null, duration = 500) {
  const [display, setDisplay] = useState<number | null>(() => {
    if (value === null) return null
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches ? value : 0
  })

  useEffect(() => {
    let frame = 0
    if (value === null) {
      frame = window.requestAnimationFrame(() => setDisplay(null))
      return () => window.cancelAnimationFrame(frame)
    }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      frame = window.requestAnimationFrame(() => setDisplay(value))
      return () => window.cancelAnimationFrame(frame)
    }
    const startedAt = performance.now()
    const tick = (now: number) => {
      const progress = clamp((now - startedAt) / duration, 0, 1)
      const eased = 1 - (1 - progress) ** 3
      setDisplay(value * eased)
      if (progress < 1) frame = window.requestAnimationFrame(tick)
    }
    frame = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(frame)
  }, [duration, value])

  return display
}

function ComprehensiveProtectionEvaluation({ evaluation }: { evaluation?: ProtectionEvaluation | null }) {
  const dimensions = completeProtectionDimensions(evaluation)
  const overallScore = optionalNumber(evaluation?.overallScore)
  const complete = evaluation?.status === 'complete' && overallScore !== null && dimensions.every((dimension) => optionalNumber(dimension.score) !== null)
  const animatedOverall = useAnimatedScore(overallScore)
  const missing = dimensions.filter((dimension) => optionalNumber(dimension.score) === null)

  return (
    <section className="relative overflow-hidden rounded-[11px] border border-cyan-300/16 bg-slate-950/16 p-5 shadow-[0_18px_60px_rgba(2,132,199,0.06)] transition duration-200 ease-out hover:-translate-y-0.5 hover:border-cyan-300/26 hover:shadow-[0_22px_70px_rgba(2,132,199,0.1)] motion-reduce:transform-none motion-reduce:transition-none">
      <div className="pointer-events-none absolute inset-x-12 top-0 h-px bg-gradient-to-r from-transparent via-cyan-200/55 to-transparent" />
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <SectionTitle>综合防护评估</SectionTitle>
            <MetricInfoButton title="综合防护评分" wide>
              <div className="space-y-5">
                <MetricFormulaContent
                  description="六项对应测试均生成后，使用加权几何平均计算综合分数。"
                  formulas={[
                    'S_{\\mathrm{overall}}=\\exp\\!\\left(\\sum_{i=1}^{6}\\alpha_i\\ln(\\max(S_i,1))\\right)',
                    '(\\alpha_1,\\ldots,\\alpha_6)=(0.20,0.10,0.20,0.15,0.15,0.20)',
                  ]}
                  note="任一必需分数缺失时不以 0 或 100 代替，综合分数保持待生成。"
                />
                <ProtectionScoreThresholdTable />
              </div>
            </MetricInfoButton>
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-500">六项分数均来自实际测试结果，分数越高表示对应防护效果越强。</p>
        </div>
        <span className={cn('rounded-full border px-3 py-1 text-xs font-black', complete ? 'border-emerald-300/20 bg-emerald-400/10 text-emerald-300' : 'border-amber-300/20 bg-amber-400/10 text-amber-200')}>
          {complete ? evaluation?.verdict || '评估已完成' : evaluation?.verdict || '待完整评估'}
        </span>
      </div>

      <div className="mt-5 grid min-h-[390px] grid-cols-[minmax(300px,1fr)_minmax(230px,0.72fr)_minmax(330px,1.12fr)] gap-5 max-xl:grid-cols-1">
        <div className="rounded-[9px] border border-cyan-300/12 bg-slate-950/24 p-3">
          <ProtectionHexRadar dimensions={dimensions} />
        </div>
        <div className="flex flex-col items-center justify-center rounded-[9px] border border-cyan-300/12 bg-[radial-gradient(circle_at_50%_18%,rgba(34,211,238,0.11),transparent_58%)] px-5 py-7 text-center">
          <p className="text-[17px] font-black tracking-[0.10em] text-cyan-100">综合防护评分</p>
          <p className={cn('mt-5 bg-gradient-to-r from-cyan-200 via-emerald-200 to-violet-200 bg-clip-text font-mono text-[42px] font-black leading-none text-transparent drop-shadow-[0_0_18px_rgba(103,232,249,0.12)]', overallScore === null && 'text-slate-500')}>
            {animatedOverall === null ? '待生成' : <MathText formula={`${animatedOverall.toFixed(2)}\\,/\\,100`} />}
          </p>
          <p className={cn('mt-4 text-xl font-black', complete ? 'text-emerald-300' : 'text-amber-200')}>{complete ? evaluation?.level : '待完整评估'}</p>
          <div className="mt-6 w-full">
            <div className="relative h-2 rounded-full bg-gradient-to-r from-rose-400 via-amber-300 to-emerald-300">
              {animatedOverall !== null ? <span className="absolute top-1/2 h-4 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/80 bg-slate-950 shadow-[0_0_10px_rgba(255,255,255,0.35)]" style={{ left: `${clamp(animatedOverall, 0, 100)}%` }} /> : null}
            </div>
            <div className="mt-1.5 flex justify-between font-mono text-[10px] text-slate-500"><span>0</span><span>50</span><span>100</span></div>
          </div>
          {missing.length ? (
            <p className="mt-5 text-xs leading-5 text-slate-500">尚缺 {missing.map((dimension) => dimension.label).join('、')}，不会以 0 或 100 代替。</p>
          ) : null}
        </div>
        <div className="rounded-[9px] border border-cyan-300/12 bg-slate-950/24 p-4">
          <h3 className="text-sm font-black text-white">六维得分明细</h3>
          <div className="mt-4 space-y-4">
            {dimensions.map((dimension) => <ProtectionDimensionRow key={dimension.key} dimension={dimension} />)}
          </div>
        </div>
      </div>
    </section>
  )
}

function MetricFormulaContent({ description, formulas, note }: { description: ReactNode; formulas: string[]; note?: ReactNode }) {
  return (
    <div className="space-y-3">
      <p>{description}</p>
      {formulas.map((formula) => (
        <MathBlock key={formula} formula={formula} className="rounded-[6px] border border-cyan-300/10 bg-cyan-400/[0.035] px-2 text-[13px] text-cyan-50" />
      ))}
      {note ? <p>{note}</p> : null}
    </div>
  )
}

function ProtectionScoreThresholdTable() {
  const rows = [
    ['WER 词错率', '<20\\%', '20\\%\\text{～}50\\%', '\\ge50\\%'],
    ['CER 字错率', '<20\\%', '20\\%\\text{～}50\\%', '\\ge50\\%'],
    ['Drift/Token 子分', '<50', '50\\text{～}80', '\\ge80'],
    ['ASR 语义保护分', '<70', '70\\text{～}85', '\\ge85'],
    ['Protected SIM', '>0.45', '0.25\\text{～}0.45', '\\le0.25'],
    ['Protected 声纹距离', '<0.55', '0.55\\text{～}0.75', '\\ge0.75'],
    ['克隆身份保护分', '<70', '70\\text{～}85', '\\ge85'],
    ['克隆语义保护分', '<70', '70\\text{～}85', '\\ge85'],
    ['克隆文本误差/变化率', '<10\\%', '10\\%\\text{～}30\\%', '\\ge30\\%'],
    ['克隆质量退化分', '<70', '70\\text{～}85', '\\ge85'],
  ]
  return (
    <div>
      <h4 className="mb-3 text-sm font-black text-cyan-100">结果解读分档参考</h4>
      <div className="overflow-x-auto rounded-[7px] border border-cyan-300/12">
        <table className="w-full min-w-[720px] border-collapse text-left text-xs leading-5">
          <thead className="bg-cyan-400/[0.08] text-cyan-100">
            <tr>
              {['内容', '偏低/较差', '中等', '较高/优秀'].map((heading) => <th key={heading} className="border-b border-cyan-300/12 px-3 py-2.5 font-black">{heading}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, weak, medium, strong]) => (
              <tr key={label} className="border-b border-cyan-300/8 last:border-0 odd:bg-white/[0.012]">
                <th className="whitespace-nowrap px-3 py-2 font-bold text-slate-200">{label}</th>
                <td className="whitespace-nowrap px-3 py-2 font-bold text-rose-700 dark:text-rose-400"><MathText formula={weak} /></td>
                <td className="whitespace-nowrap px-3 py-2 font-bold text-amber-700 dark:text-amber-400"><MathText formula={medium} /></td>
                <td className="whitespace-nowrap px-3 py-2 font-bold text-emerald-700 dark:text-emerald-400"><MathText formula={strong} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ProtectionDimensionRow({ dimension }: { dimension: ProtectionEvaluationDimension }) {
  const score = optionalNumber(dimension.score)
  const animatedScore = useAnimatedScore(score)
  return (
    <MetricInfoSurface
      title={dimension.label || protectionDimensionLabels[dimension.key]}
      info={protectionDimensionInfo[dimension.key]}
      tooltip={dimension.reason ? friendlyCloneMetricReason(dimension.reason) : undefined}
      className="rounded-[7px] border border-transparent px-2 py-1.5"
    >
      <div className="mb-1.5 flex items-center gap-2 text-xs">
        <span className={cn('grid h-6 w-6 shrink-0 place-items-center rounded-[6px]', score === null ? 'bg-slate-800 text-slate-500' : 'bg-cyan-400/10 text-cyan-300')}>{protectionDimensionIcon(dimension.key)}</span>
        <span className="min-w-0 flex-1 truncate font-bold text-slate-300">{dimension.label || protectionDimensionLabels[dimension.key]}</span>
        <span className={cn('shrink-0 font-mono font-black', score === null ? 'text-slate-500' : 'text-cyan-200')}>{animatedScore === null ? '待生成' : <MathText formula={`${animatedScore.toFixed(2)}\\,/\\,100`} />}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
        {animatedScore !== null ? <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-sky-300 to-emerald-300" style={{ width: `${clamp(animatedScore, 0, 100)}%` }} /> : null}
      </div>
    </MetricInfoSurface>
  )
}

function ProtectionHexRadar({ dimensions }: { dimensions: ProtectionEvaluationDimension[] }) {
  const themeMode = useAppStore((state) => state.themeMode)
  const labelColor = themeMode === 'light' ? '#0f172a' : '#cbd5e1'
  const missingColor = themeMode === 'light' ? '#1e293b' : '#64748b'
  const valueColor = themeMode === 'light' ? '#0c4a6e' : '#38bdf8'
  const width = 440
  const height = 330
  const centerX = width / 2
  const centerY = height / 2
  const radius = 108
  const scores = dimensions.map((dimension) => optionalNumber(dimension.score))
  const ready = scores.every((score): score is number => score !== null)
  const axes = dimensions.map((dimension, index) => {
    const angle = -Math.PI / 2 + (index / dimensions.length) * Math.PI * 2
    return {
      dimension,
      angle,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
      labelX: centerX + Math.cos(angle) * (radius + 52),
      labelY: centerY + Math.sin(angle) * (radius + 32),
    }
  })
  const polygon = ready
    ? axes.map((axis, index) => {
        const normalized = clamp(scores[index] / 100, 0, 1)
        return `${(centerX + Math.cos(axis.angle) * radius * normalized).toFixed(1)},${(centerY + Math.sin(axis.angle) * radius * normalized).toFixed(1)}`
      }).join(' ')
    : ''

  return (
    <div className="relative grid h-full min-h-[350px] place-items-center overflow-hidden">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full max-w-[500px]">
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <polygon key={scale} points={axes.map((axis) => `${(centerX + Math.cos(axis.angle) * radius * scale).toFixed(1)},${(centerY + Math.sin(axis.angle) * radius * scale).toFixed(1)}`).join(' ')} fill="none" stroke="rgba(148,163,184,.16)" strokeWidth="1" />
        ))}
        {axes.map((axis) => <line key={axis.dimension.key} x1={centerX} y1={centerY} x2={axis.x} y2={axis.y} stroke="rgba(148,163,184,.18)" strokeWidth="1" />)}
        {ready ? (
          <g className="origin-center motion-safe:animate-[pulse_500ms_ease-out_1] motion-reduce:animate-none" style={{ transformOrigin: `${centerX}px ${centerY}px` }}>
            <polygon points={polygon} fill="rgba(34,211,238,.17)" stroke="#67e8f9" strokeWidth="2" />
            {axes.map((axis, index) => {
              const normalized = clamp(scores[index] / 100, 0, 1)
              return <circle key={axis.dimension.key} cx={centerX + Math.cos(axis.angle) * radius * normalized} cy={centerY + Math.sin(axis.angle) * radius * normalized} r="3.5" fill="#a7f3d0" />
            })}
          </g>
        ) : null}
        {axes.map((axis) => (
          <g key={`label-${axis.dimension.key}`}>
            <text x={axis.labelX} y={axis.labelY} textAnchor={axis.labelX < centerX - 15 ? 'end' : axis.labelX > centerX + 15 ? 'start' : 'middle'} fontSize="11.5" fontWeight="850" fill={labelColor}>{protectionRadarLabels[axis.dimension.key]}</text>
            <text x={axis.labelX} y={axis.labelY + 17} textAnchor={axis.labelX < centerX - 15 ? 'end' : axis.labelX > centerX + 15 ? 'start' : 'middle'} fontSize="12" fontWeight="850" fill={optionalNumber(axis.dimension.score) === null ? missingColor : valueColor}>{optionalNumber(axis.dimension.score) === null ? '待生成' : optionalNumber(axis.dimension.score)?.toFixed(2)}</text>
          </g>
        ))}
      </svg>
      {!ready ? <div className="pointer-events-none absolute inset-x-1/4 top-1/2 -translate-y-1/2 rounded-[7px] border border-dashed border-cyan-300/14 bg-slate-950/90 px-3 py-2 text-center text-xs leading-5 text-slate-400">完成六项对应测试后生成雷达图</div> : null}
    </div>
  )
}

function asrHistoryLifecycleStatus(item?: AsrHistoryEntry) {
  return String(item?.taskStatus?.status ?? item?.status ?? item?.asr?.status ?? '').toLowerCase()
}

function asrHistoryFailureReason(item: AsrHistoryEntry) {
  const snapshotError = item.taskStatus?.error
  const snapshotMessage = typeof snapshotError === 'string' ? snapshotError : snapshotError?.message
  return friendlyAsrFailure(item.asr?.error || item.asr?.reason || snapshotMessage || item.taskStatus?.message)
}

function asrHistoryProgress(item: AsrHistoryEntry) {
  const status = asrHistoryLifecycleStatus(item)
  const stored = optionalNumber(item.taskStatus?.progress)
  const terminal = ['completed', 'success', 'failed', 'error', 'cancelled', 'available', 'computed', 'partial', 'unavailable'].includes(status)
  return clamp(stored ?? (terminal || item.asr ? 1 : 0), 0, 1)
}

function asrHistoryStatusLabel(item: AsrHistoryEntry) {
  const status = asrHistoryLifecycleStatus(item)
  if (status === 'queued') return '等待中'
  if (status === 'running') return '进行中'
  if (status === 'failed' || status === 'error' || status === 'unavailable') return '失败'
  if (status === 'cancelled') return '已取消'
  return '已完成'
}

function cloneHistoryLifecycleStatus(item?: CloneHistoryEntry) {
  return String(item?.taskStatus?.status ?? item?.status ?? item?.result?.status ?? '').toLowerCase()
}

function cloneHistoryFailureReason(item: CloneHistoryEntry) {
  const snapshotError = item.taskStatus?.error
  const snapshotMessage = typeof snapshotError === 'string' ? snapshotError : snapshotError?.message
  return shortMetricReason(snapshotMessage || item.taskStatus?.message || item.result?.message || '克隆任务未生成可用结果。')
}

function cloneHistoryProgress(item: CloneHistoryEntry) {
  const status = cloneHistoryLifecycleStatus(item)
  const stored = optionalNumber(item.taskStatus?.progress)
  const terminal = ['completed', 'success', 'failed', 'error', 'cancelled', 'available', 'computed', 'partial', 'unavailable'].includes(status)
  return clamp(stored ?? (terminal || item.result ? 1 : 0), 0, 1)
}

function lifecycleStatusLabel(statusValue?: string) {
  const status = String(statusValue ?? '').toLowerCase()
  if (status === 'queued') return '等待中'
  if (status === 'running') return '进行中'
  if (status === 'partial_failed') return '部分失败'
  if (status === 'failed' || status === 'error' || status === 'unavailable') return '失败'
  if (status === 'cancelled') return '已取消'
  return '已完成'
}

function progressTone(statusValue?: string) {
  const status = String(statusValue ?? '').toLowerCase()
  if (status === 'failed' || status === 'error' || status === 'unavailable' || status === 'partial_failed' || status === 'cancelled') {
    return { fill: 'bg-rose-400', text: 'text-rose-300' }
  }
  if (status === 'running') return { fill: 'bg-amber-400', text: 'text-amber-300' }
  if (status === 'queued') return { fill: 'bg-cyan-400', text: 'text-cyan-300' }
  return { fill: 'bg-emerald-400', text: 'text-emerald-300' }
}

function batchElapsed(batch: EvaluationBatch) {
  const stored = optionalNumber(batch.elapsedSec)
  if (stored !== null) return stored
  const childElapsed = batch.items.map((item) => optionalNumber(item.elapsedSec)).filter((value): value is number => value !== null)
  return childElapsed.length ? Math.max(...childElapsed) : null
}

function AsrTab({ result, asrEval, editStats, history, batches = [], selectedAsrSubId, onSelect, onOpenBatch, historyLimit = 5 }: { result: TaskResult; asrEval?: AsrEval | null; editStats: EditMetrics | null; history: AsrHistoryEntry[]; batches?: EvaluationBatch[]; selectedAsrSubId?: string; onSelect: (asrSubId?: string) => void; onOpenBatch: (batch: EvaluationBatch) => void; historyLimit?: number }) {
  const selectedHistory = selectedAsrSubId ? history.find((item) => item.asrSubId === selectedAsrSubId) : history.at(-1)
  if (!asrEval) {
    const lifecycleStatus = asrHistoryLifecycleStatus(selectedHistory) || (selectedAsrSubId ? 'queued' : '')
    const failed = lifecycleStatus === 'failed' || lifecycleStatus === 'error' || lifecycleStatus === 'unavailable'
    const pending = lifecycleStatus === 'queued' || lifecycleStatus === 'running'
    return (
      <div className="space-y-5">
        <AsrHistoryPanel history={history} batches={batches} selectedAsrSubId={selectedAsrSubId} onSelect={onSelect} onOpenBatch={onOpenBatch} maxVisible={historyLimit} />
        <EmptyState
          title={failed ? 'ASR 测试失败' : pending ? lifecycleStatus === 'queued' ? 'ASR 测试等待中' : 'ASR 测试进行中' : '未执行 ASR 测试'}
          text={failed && selectedHistory ? asrHistoryFailureReason(selectedHistory) : selectedHistory?.taskStatus?.message || (pending ? '任务状态会自动刷新，完成后显示完整转写与指标。' : '开始 ASR 测试后，可查看识别文本差异和表达内容保护指标。')}
        />
      </div>
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
  const sharedSemantic = result.semanticEval
  const tokenDiff = sharedSemantic?.tokenChangeRate ?? sharedSemantic?.tokenErrorRate
  const tokenUsesEditDistance = sharedSemantic?.tokenChangeRate == null && sharedSemantic?.tokenErrorRate != null
  const tokenDetailLabel = tokenUsesEditDistance ? 'Token 编辑率' : 'Token 变化率'
  const tokenUnavailableReason = tokenDiff == null
    ? sharedSemantic?.error || sharedSemantic?.reason || metricReason(result, ['semanticEval.tokenChangeRate', 'semanticEval.tokenErrorRate', 'asrEval.tokenChangeRate', 'asrEval.tokenErrorRate'])
    : ''
  const tokenFoot = tokenUsesEditDistance ? (
    <MetricFormulaContent
      description={`${tokenUnavailableReason ? `${tokenUnavailableReason}。` : ''}当前回退值使用保护前后离散语音 Token 序列的编辑距离，并除以原始序列长度。`}
      formulas={['R_{\\mathrm{token}}=\\frac{D_{\\mathrm{edit}}(z,z^{\\prime})}{\\max(|z|,1)}']}
      note="该指标不按 ASR 文本的字符或单词切分。"
    />
  ) : (
    <MetricFormulaContent
      description={`${tokenUnavailableReason ? `${tokenUnavailableReason}。` : ''}将保护前后的音频转换为离散语音 Token，统计两侧较短序列内同位置 Token 不同的比例。`}
      formulas={['R_{\\mathrm{token}}=\\frac{1}{\\max(L,1)}\\sum_{i=1}^{L}\\mathbf{1}[z_i\\ne z_i^{\\prime}],\\qquad L=\\min(|z|,|z^{\\prime}|)']}
      note="该指标不按 ASR 文本的字符或单词切分。"
    />
  )
  const semanticSourceInfo = metricSource(result, ['semanticEval.semanticDrift', 'asrEval.semanticDrift'])
  const semanticIsMfccProxy = String(semanticSourceInfo?.source ?? '').toLowerCase() === 'mfcc_proxy'
  const semanticUnavailableReason = sharedSemantic?.semanticDrift == null
    ? sharedSemantic?.error || sharedSemantic?.reason || metricReason(result, ['semanticEval.semanticDrift', 'asrEval.semanticDrift'])
    : ''
  const semanticFoot = semanticIsMfccProxy ? (
    <MetricFormulaContent
      description={semanticUnavailableReason || '比较保护前后音频的 MFCC 声学特征变化。'}
      formulas={["D_{\\mathrm{sem}}=1-\\overline{\\cos(F(x),F(x^{\\prime}))}"]}
      note="当前为 MFCC 声学特征代理，不等同于深度语义表示。"
    />
  ) : (
    <MetricFormulaContent
      description={semanticUnavailableReason || '综合不同语义表示层的余弦距离，衡量保护前后的语义表示变化。'}
      formulas={["D_{\\mathrm{sem}}=\\frac{\\sum_k w_k\\left(1-\\operatorname{mean}_t\\cos(F_k(x)_t,F_k(x^{\\prime})_t)\\right)}{\\sum_k w_k}"]}
      note="数值越高，表示语义表示变化越大。"
    />
  )
  const semanticDetailLabel = semanticIsMfccProxy ? 'MFCC 代理漂移' : '语义表示漂移'
  const errorShares = resolveAsrErrorShares(asrEval.errorShares, asrEval.editCounts, editStats?.errorShares)
  const asrFailureReason = ['unavailable', 'failed', 'error'].includes(String(asrEval.status ?? '').toLowerCase())
    ? friendlyAsrFailure(asrEval.error || asrEval.reason)
    : null

  return (
    <div className="space-y-5">
      <AsrHistoryPanel
        history={history}
        batches={batches}
        selectedAsrSubId={selectedAsrSubId}
        onSelect={onSelect}
        onOpenBatch={onOpenBatch}
        maxVisible={historyLimit}
        activeLabel={`ASR 标注 · ${shortAsrModelName(asrEval.model)}`}
      />
      {asrFailureReason ? <MetricNotice text={`该次 ASR 转写失败：${asrFailureReason}`} /> : null}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_288px_minmax(0,1fr)]">
          <TextBox title="原始音频ASR自动标注结果" text={referenceText || '未生成'} foot="作为识别错误率和文本差异的参考文本" />
        <div className="grid grid-cols-2 content-center gap-3">
          <ScoreBox label={<>WER<br />词错率</>} value={formatAsrRatePercent(wer)} red compact foot={<MetricFormulaContent description="词级识别错误率。" formulas={['\\mathrm{WER}=\\frac{S_w+D_w+I_w}{\\max(N_w,1)}']} note="其中 S、D、I 分别为词级替换、删除和插入数量，N 为参考文本词数。" />} />
          <ScoreBox label={<>CER<br />字错率</>} value={formatAsrRatePercent(cer)} red compact foot={<MetricFormulaContent description="字符级识别错误率。" formulas={['\\mathrm{CER}=\\frac{S_c+D_c+I_c}{\\max(N_c,1)}']} note="其中 S、D、I 分别为字符级替换、删除和插入数量，N 为参考文本字符数。" />} />
          <ScoreBox label={<>IR<br />插入率</>} value={formatAsrRatePercent(insertRate)} red compact foot={<MetricFormulaContent description="衡量识别结果相对参考文本新增内容的比例。" formulas={['\\mathrm{IR}=\\frac{I}{\\max(N,1)}']} note="其中 I 为新增单位数，N 为参考文本在当前统计层级下的单位数。" />} />
          <ScoreBox label={<>SR<br />替换率</>} value={formatAsrRatePercent(substituteRate)} red compact foot={<MetricFormulaContent description="衡量识别结果相对参考文本发生替换的比例。" formulas={['\\mathrm{SR}=\\frac{S}{\\max(N,1)}']} note="其中 S 为替换单位数，N 为参考文本在当前统计层级下的单位数。" />} />
        </div>
        <TextBox title="保护音频ASR自动标注结果" text={protectedText || '未生成'} foot="红色为新增内容，绿色删除线为原文缺失内容" content={diffOps.length ? renderDiffOps(diffOps) : undefined} />
      </div>
      <div className="grid grid-cols-[minmax(300px,1fr)_minmax(0,2fr)] items-stretch gap-5 max-lg:grid-cols-1">
        <div className="min-h-0 space-y-5">
          <MetricPanel title="保护任务共享语义指标">
            <ScoreBox label={semanticDetailLabel} value={formatMetricValue(sharedSemantic?.semanticDrift, 'number')} foot={semanticFoot} />
            <ScoreBox label={tokenDetailLabel} value={formatAsrRatePercent(tokenDiff)} foot={tokenFoot} />
          </MetricPanel>
          <RateBreakdown substituteShare={errorShares?.substituteShare} insertShare={errorShares?.insertShare} />
        </div>
        <div className="relative min-h-0 max-lg:min-h-[320px]">
          <div className="absolute inset-0 max-lg:static max-lg:h-[360px]">
            <InsightPanel
              title="ASR 保护结果解读"
              items={generateAsrMetricInsights(asrEval, editStats, sharedSemantic ? { ...sharedSemantic, semanticIsMfccProxy } : null)}
              fillHeight
            />
          </div>
        </div>
      </div>
    </div>
  )
}

function CloneTab({ result, cloneResult, cloneEval, cloneHistory, cloneBatches, selectedCloneKey, onSelectClone, onOpenAsr, loading, status, cloneModelOptions, modelTypes, asrEval, asrEditStats, asrHistory, asrBatches, selectedAsrSubId, asrHistoryLimit, cloneHistoryLimit }: { result: TaskResult; cloneResult?: CloneVoiceResult; cloneEval?: CloneEval | null; cloneHistory: CloneHistoryEntry[]; cloneBatches: EvaluationBatch[]; selectedCloneKey?: string; onSelectClone: (cloneKey: string) => void; onOpenAsr: (asrSubId?: string) => void; loading: boolean; status: TaskStatusResponse | null; cloneModelOptions: BackendSelectOption[]; modelTypes?: CapabilitiesResponse['modelTypes']; asrEval?: AsrEval | null; asrEditStats: EditMetrics | null; asrHistory: AsrHistoryEntry[]; asrBatches: EvaluationBatch[]; selectedAsrSubId?: string; asrHistoryLimit: number; cloneHistoryLimit: number }) {
  const [manualAnnotation, setManualAnnotation] = useState<CloneHistoryEntry | null>(null)
  const [informationModel, setInformationModel] = useState<BackendSelectOption | null>(null)
  const [batchDetail, setBatchDetail] = useState<EvaluationBatch | null>(null)
  const openBatch = (batch: EvaluationBatch) => {
    setManualAnnotation(null)
    setBatchDetail(batch)
  }
  const liveBatchDetail = batchDetail
    ? (batchDetail.type === 'asr' ? asrBatches : cloneBatches).find((item) => item.batchId === batchDetail.batchId) ?? batchDetail
    : null
  const selectedCloneEntry = selectedCloneKey ? cloneHistory.find((item) => item.key === selectedCloneKey) : undefined
  const selectedCloneStatus = String(selectedCloneEntry?.taskStatus?.status ?? selectedCloneEntry?.status ?? '').toLowerCase()
  const selectedClonePending = selectedCloneStatus === 'queued' || selectedCloneStatus === 'running'
  const selectedCloneFailed = ['failed', 'error', 'cancelled', 'unavailable'].includes(selectedCloneStatus)
  const selectedCloneError = selectedCloneEntry ? cloneHistoryFailureReason(selectedCloneEntry) : null
  const activeCloneRequest = selectedCloneEntry?.request ?? cloneResult?.request
  const activeCloneModelValue = cloneEval?.cloneModel ?? activeCloneRequest?.model
  const activeCloneModelOption = cloneModelOptions.find((item) => item.value === activeCloneModelValue || item.backendValue === activeCloneModelValue) ?? null
  const activeCloneAnnotation = cloneAnnotationTitle(activeCloneRequest, activeCloneModelOption, cloneResult?.fineTune)
  const asrSection = asrEval || asrHistory.length || asrBatches.length || selectedAsrSubId ? (
    <div id="asr-result-detail" className="scroll-mt-24">
      <AsrTab result={result} asrEval={asrEval} editStats={asrEditStats} history={asrHistory} batches={asrBatches} selectedAsrSubId={selectedAsrSubId} onSelect={onOpenAsr} onOpenBatch={openBatch} historyLimit={asrHistoryLimit} />
    </div>
  ) : null
  const cloneHistorySection = cloneHistory.length || cloneBatches.length ? (
    <CloneHistoryPanel
      history={cloneHistory}
      batches={cloneBatches}
      modelOptions={cloneModelOptions}
      selectedCloneKey={selectedCloneKey}
      onSelect={onSelectClone}
      onOpenBatch={openBatch}
      onOpenAsr={onOpenAsr}
      onOpenManual={(item) => { setBatchDetail(null); setManualAnnotation(item) }}
      maxVisible={cloneHistoryLimit}
      activeLabel={cloneEval ? activeCloneAnnotation ?? undefined : undefined}
      activeModel={cloneEval ? shortCloneModelName(activeCloneModelValue) : undefined}
      activeModelOption={activeCloneModelOption}
      onOpenModel={setInformationModel}
    />
  ) : null
  const showLoading = loading || selectedClonePending

  let cloneDetail: ReactNode
  if (showLoading) {
    const liveStatus = selectedCloneEntry?.taskStatus ?? status
    cloneDetail = (
      <div className="grid items-center gap-6 pl-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
        <LoadingCard title="原始克隆语音" progress={optionalNumber(liveStatus?.progress) ?? undefined} message={liveStatus?.message ?? undefined} />
        <div className="compare-badge mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
        <LoadingCard title="保护后克隆语音" progress={optionalNumber(liveStatus?.progress) ?? undefined} message={liveStatus?.message ?? undefined} />
      </div>
    )
  } else if (selectedCloneFailed) {
    cloneDetail = <EmptyState title={selectedCloneStatus === 'cancelled' ? '语音克隆测试已取消' : '语音克隆测试失败'} text={selectedCloneError || '该次克隆任务未生成可用结果。'} />
  } else if (!cloneEval) {
    cloneDetail = <EmptyState title="未执行语音克隆测试" text="请在防护工作台选择已完成的保护任务并开始语音克隆测试。" />
  } else {
    const cloneReason = cloneEval.reason ? shortMetricReason(cloneEval.reason) : ''
    cloneDetail = (
      <>
        <div id="clone-result-detail" className="scroll-mt-24 space-y-5">
          {cloneReason ? <MetricNotice text={`克隆指标未生成原因：${friendlyCloneMetricReason(cloneReason)}`} /> : null}
        <div className="grid items-center gap-6 pl-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
          {cloneEval.originalCloneAudio ? <AudioCard title="原始克隆语音" audio={cloneEval.originalCloneAudio} color="#a78bfa" compactHeader /> : <EmptyMetricCard title="原始克隆语音" text="暂未生成原始克隆语音" />}
          <div className="compare-badge mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
          {cloneEval.protectedCloneAudio ? <AudioCard title="保护后克隆语音" audio={cloneEval.protectedCloneAudio} color="#f59e0b" compactHeader /> : <EmptyMetricCard title="保护后克隆语音" text="暂未生成保护后克隆语音" />}
        </div>
        <CloneCoreMetricCards cloneEval={cloneEval} />
        <CloneSemanticExpressionPanel cloneEval={cloneEval} insights={generateCloneInsights(cloneEval)} />
        </div>
      </>
    )
  }

  return (
    <div className="space-y-5">
      {asrSection}
      {cloneHistorySection}
      {cloneDetail}
      <ManualAnnotationModal item={manualAnnotation} onClose={() => setManualAnnotation(null)} />
      <BatchProgressModal batch={liveBatchDetail} cloneModelOptions={cloneModelOptions} onClose={() => setBatchDetail(null)} />
      <ModelInformationModal model={informationModel} modelTypes={modelTypes} onClose={() => setInformationModel(null)} />
    </div>
  )
}

function LoadingCard({ title, progress, message }: { title: string; progress?: number; message?: string }) {
  const normalizedProgress = progress === undefined ? undefined : clamp(progress, 0, 1)
  return (
    <div className="grid h-[252px] place-items-center rounded-[9px] border border-violet-300/18 bg-violet-400/8 p-5 text-center">
      <div className="w-full">
        <Loader2 className="mx-auto h-9 w-9 animate-spin text-violet-200" />
        <p className="mt-4 text-sm font-black text-slate-100">{title}</p>
        {normalizedProgress !== undefined ? (
          <div className="mt-3 mx-auto max-w-[180px]">
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-violet-400 transition-all duration-500" style={{ width: `${Math.round(normalizedProgress * 100)}%` }} />
            </div>
            <p className="mt-1 font-mono text-[10px] text-slate-400">{Math.round(normalizedProgress * 100)}%</p>
          </div>
        ) : null}
        {message ? <p className="mt-2 text-xs text-slate-400">{message}</p> : <p className="mt-2 text-xs text-slate-400">正在等待克隆音频...</p>}
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
  const info = (
    <MetricFormulaContent
      description="分别统计替换与插入在全部编辑错误中的占比，用于判断保护音频主要通过哪类文本变化干扰识别。"
      formulas={[
        'p_{\\mathrm{sub}}=\\frac{S}{\\max(S+D+I,1)}',
        'p_{\\mathrm{ins}}=\\frac{I}{\\max(S+D+I,1)}',
      ]}
      note="分母中的删除数量仍参与总错误数计算；页面只单独展示替换与插入两类占比。"
    />
  )
  return (
    <MetricInfoSurface title="错误类型占比" info={info} className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <SectionTitle>错误类型占比</SectionTitle>
      <div className="mt-5 space-y-4">
        {rows.map(([label, value, color]) => {
          const numberValue = optionalNumber(value)
          return (
            <div key={label}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-bold text-slate-300">{label}</span>
                <span className="font-mono text-slate-400">{formatAsrRatePercent(numberValue)}</span>
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
    </MetricInfoSurface>
  )
}

function CloneCoreMetricCards({ cloneEval }: { cloneEval: CloneEval }) {
  const before = optionalNumber(cloneEval.embeddingDistanceBefore)
  const after = optionalNumber(cloneEval.embeddingDistanceAfter)
  const delta = optionalNumber(cloneEval.embeddingDistanceDelta) ?? computeAbsoluteDelta(before, after)
  return (
    <div className="grid grid-cols-4 gap-3 max-lg:grid-cols-2 max-sm:grid-cols-1">
      <CloneDistanceCoreCard before={before} after={after} delta={delta} reason={cloneEval.cloneIdentityReason ?? cloneEval.reason} />
      <CloneScoreCoreCard
        title="克隆身份保护评分"
        score={optionalNumber(cloneEval.cloneIdentityScore)}
        reason={cloneEval.cloneIdentityReason ?? cloneEval.reason}
        detail="声音身份分离效果"
        icon={<Fingerprint className="h-4 w-4" />}
        tone="cyan"
        info={(
          <MetricFormulaContent
            description="针对当前选中的单个克隆结果，先比较保护前后的声纹距离，再转换为单模型身份保护分数。"
            formulas={[
              "d=1-\\operatorname{SIM}(x,c),\\qquad d^{\\prime}=1-\\operatorname{SIM}(x,c^{\\prime})",
              "P=\\operatorname{clip}\\!\\left(\\frac{d^{\\prime}-d}{0.75-d},0,1\\right)",
              "B=5\\,\\operatorname{clip}\\!\\left(\\frac{d^{\\prime}-0.75}{0.25},0,1\\right)",
              'S_{\\mathrm{clone\\_id}}=95P+B',
            ]}
            note="分数越高，表示保护后克隆声音越难保持原说话人的声音身份。"
          />
        )}
      />
      <CloneScoreCoreCard
        title="克隆后语义干扰评分"
        score={optionalNumber(cloneEval.cloneSemanticScore)}
        reason={cloneEval.cloneSemanticReason ?? cloneEval.cloneAsrReason}
        detail={optionalNumber(cloneEval.cloneTextChangeRate) === null ? '表达内容变化效果' : `文本变化 ${formatMetricValue(cloneEval.cloneTextChangeRate, 'percent')}`}
        icon={<BrainCircuit className="h-4 w-4" />}
        tone="violet"
        info={(
          <MetricFormulaContent
            description="针对当前选中的单个克隆结果，使用离散语音 Token 变化与语义表示漂移共同计算分数。"
            formulas={[
              '\\Phi(x;x_{90})=100\\left(1-10^{-x/x_{90}}\\right)',
              'S_{\\mathrm{clone\\_sem}}=0.55\\,\\Phi(R_{\\mathrm{clone\\_token}};R_{90}^{\\mathrm{clone}})+0.45\\,\\Phi(D_{\\mathrm{clone\\_sem}};D_{90}^{\\mathrm{clone}})',
            ]}
            note="分数越高，表示保护后克隆语音的表达内容受到的干扰越明显。"
          />
        )}
      />
      <CloneScoreCoreCard
        title="克隆音频质量退化评分"
        score={optionalNumber(cloneEval.cloneQualityScore)}
        reason={cloneEval.cloneQualityReason}
        detail={optionalNumber(cloneEval.cloneQualityDropRate) === null ? '语音质量变化效果' : `质量下降 ${formatMetricValue(cloneEval.cloneQualityDropRate, 'percent')}`}
        icon={<Waves className="h-4 w-4" />}
        tone="amber"
        info={(
          <MetricFormulaContent
            description="以原音频生成的克隆音频为参考，与保护音频生成的克隆音频比较 PESQ、STOI 和 DNSMOS，衡量克隆后的听感下降。"
            formulas={[
              "Q=0.45S_{\\mathrm{PESQ}}+0.45S_{\\mathrm{STOI}}+0.10S_{\\mathrm{DNSMOS}}",
              "Q^{\\prime}=0.45S_{\\mathrm{PESQ}}^{\\prime}+0.45S_{\\mathrm{STOI}}^{\\prime}+0.10S_{\\mathrm{DNSMOS}}^{\\prime}",
              "d_q=\\max\\!\\left(0,\\frac{Q-Q^{\\prime}}{Q}\\right)",
              'S_q^{\\mathrm{raw}}=\\Phi(d_q;0.75)',
            ]}
            note="得到听感下降分数后再对身份和语义分数进行综合考虑，给出最终的质量下降分数。"
          />
        )}
      />
    </div>
  )
}

function CloneCoreCardHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('flex min-h-9 items-center gap-2 text-[14px] font-black leading-5', className)}>{children}</div>
  )
}

function CloneDistanceCoreCard({ before, after, delta, reason }: { before: number | null; after: number | null; delta: number | null; reason?: string | null }) {
  const available = before !== null && after !== null
  const info = (
    <MetricFormulaContent
      description="ECAPA-TDNN 是用于表示说话人声音身份的声纹模型。这里分别计算原音频与两段克隆语音的声纹距离。"
      formulas={[
        '\\operatorname{SIM}(a,b)=\\cos(\\operatorname{Emb}(a),\\operatorname{Emb}(b))',
        "d=1-\\operatorname{SIM}(x,c),\\qquad d^{\\prime}=1-\\operatorname{SIM}(x,c^{\\prime})",
        "\\Delta d=d^{\\prime}-d",
      ]}
      note="右侧距离越大，表示保护后的克隆声音越远离原说话人。"
    />
  )
  const deltaText = delta === null
    ? '变化未生成'
    : delta > 0
      ? `身份距离增大 ${delta.toFixed(2)}`
      : delta < 0
        ? `身份距离减小 ${Math.abs(delta).toFixed(2)}`
        : '身份距离无明显变化'
  return (
    <MetricInfoSurface title="指标信息" info={info} tooltip={!available ? friendlyCloneMetricReason(reason) : undefined} className="relative min-h-[154px] overflow-hidden rounded-[9px] border border-cyan-300/14 bg-slate-950/22 p-4">
      <CloneCoreCardHeader className="text-cyan-100"><Fingerprint className="h-4 w-4 text-cyan-300" />克隆后身份差异（核心）</CloneCoreCardHeader>
      {available ? (
        <>
          <div className="mt-5 flex items-center justify-center gap-3 font-mono text-[34px] font-black leading-none"><span className="text-slate-100">{before.toFixed(2)}</span><MathText formula={'\\longrightarrow'} className="text-[33px] text-slate-400" /><span className="text-emerald-400">{after.toFixed(2)}</span></div>
          <p className={cn('mt-4 text-center text-[15px] font-black', (delta ?? 0) >= 0 ? 'text-emerald-300' : 'text-rose-300')}>{deltaText}</p>
        </>
      ) : <p className="mt-7 text-center text-sm font-bold text-slate-500">待生成</p>}
    </MetricInfoSurface>
  )
}

function CloneScoreCoreCard({ title, score, reason, detail, icon, tone, info }: { title: string; score: number | null; reason?: string | null; detail: string; icon: ReactNode; tone: 'cyan' | 'violet' | 'amber'; info: ReactNode }) {
  const animatedScore = useAnimatedScore(score)
  const toneClass = tone === 'violet' ? 'text-violet-300' : tone === 'amber' ? 'text-amber-300' : 'text-cyan-300'
  const barClass = tone === 'violet' ? 'from-violet-400 to-fuchsia-300' : tone === 'amber' ? 'from-amber-400 to-orange-300' : 'from-cyan-400 to-emerald-300'
  return (
    <MetricInfoSurface title={title} info={info} tooltip={score === null ? friendlyCloneMetricReason(reason) : undefined} className="min-h-[154px] rounded-[9px] border border-cyan-300/14 bg-slate-950/22 p-4">
      <CloneCoreCardHeader className="text-slate-100"><span className={toneClass}>{icon}</span>{title}</CloneCoreCardHeader>
      <p className={cn('mt-5 text-center font-mono text-[30px] font-black leading-none', score === null ? 'text-slate-500' : toneClass)}>{animatedScore === null ? '待生成' : `${animatedScore.toFixed(2)} 分`}</p>
      <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-800">{animatedScore !== null ? <div className={cn('h-full rounded-full bg-gradient-to-r', barClass)} style={{ width: `${clamp(animatedScore, 0, 100)}%` }} /> : null}</div>
      <p className="mt-2 truncate text-center text-[11px] text-slate-500">{score === null ? friendlyCloneMetricReason(reason) : detail}</p>
    </MetricInfoSurface>
  )
}

function CloneSemanticExpressionPanel({ cloneEval, insights }: { cloneEval: CloneEval; insights: string[] }) {
  const targetText = cloneEval.targetText ?? ''
  const cleanText = cloneEval.cleanCloneTranscription ?? ''
  const protectedText = cloneEval.protectedCloneTranscription ?? ''
  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><SectionTitle>克隆后表达的实际语义</SectionTitle><p className="mt-2 text-xs text-slate-500">查看两段克隆语音最终表达了什么，以及保护前后的文本变化。</p></div>
        {cloneEval.cloneAsrStatus && !metricStatusAvailable(cloneEval.cloneAsrStatus) ? <span className="rounded-full border border-amber-300/18 bg-amber-400/8 px-2.5 py-1 text-[11px] font-bold text-amber-200">文本尚未完整生成</span> : null}
      </div>
      <div className="mt-4 grid grid-cols-3 gap-3 max-lg:grid-cols-1">
        <CloneTranscriptColumn title="目标文本（输入）" text={targetText} />
        <CloneTranscriptColumn title="原始克隆语音文本（自动转写）" text={cleanText} reference={targetText} />
        <CloneTranscriptColumn title="保护后克隆语音文本（自动转写）" text={protectedText} reference={targetText} />
      </div>
      <div className="mt-4 grid grid-cols-[minmax(180px,1fr)_minmax(0,5fr)] items-stretch gap-5 max-lg:grid-cols-1">
        <div className="grid content-start gap-3">
          <CloneTextMetric label="原始克隆文本误差" value={cloneEval.cleanCloneTextError} />
          <CloneTextMetric label="保护后克隆文本误差" value={cloneEval.protectedCloneTextError} />
          <CloneTextMetric label="文本变化率" value={cloneEval.cloneTextChangeRate} />
        </div>
        <div className="relative min-h-0 max-lg:min-h-[320px]">
          <div className="absolute inset-0 max-lg:static max-lg:h-[360px]">
            <InsightPanel title="克隆结果解读" items={insights} fillHeight />
          </div>
        </div>
      </div>
      {(!cleanText || !protectedText) && cloneEval.cloneAsrReason ? <p className="mt-3 text-xs leading-5 text-slate-500">未生成原因：{friendlyCloneMetricReason(cloneEval.cloneAsrReason)}</p> : null}
    </section>
  )
}

function CloneTranscriptColumn({ title, text, reference }: { title: string; text: string; reference?: string }) {
  const editMetrics = text && reference ? computeEditMetrics(reference, text, chooseEditLevel(reference, text)) : null
  const content = editMetrics ? renderDiffOps(editMetrics.diffOps) : text
  return (
    <div className="min-w-0 rounded-[9px] border border-cyan-300/12 bg-slate-950/22 p-4">
      <h3 className="text-center text-sm font-black text-cyan-100">{title}</h3>
      <div className="mt-3 h-[156px] min-w-0 overflow-x-hidden overflow-y-auto whitespace-pre-wrap break-words rounded-[7px] border border-cyan-300/8 bg-slate-950/30 p-3 text-sm leading-7 text-slate-200 [overflow-wrap:anywhere]">{content || <span className="text-slate-500">暂未生成</span>}</div>
    </div>
  )
}

function CloneTextMetric({ label, value }: { label: string; value?: number | null }) {
  const numberValue = optionalNumber(value)
  return (
    <div className="rounded-[8px] border border-cyan-300/10 bg-slate-950/22 px-4 py-4 text-center">
      <p className="text-xs font-bold text-slate-400">{label}</p>
      <p className={cn('mt-2 font-mono text-xl font-black', numberValue === null ? 'text-slate-500' : 'text-cyan-200')}>{numberValue === null ? '待生成' : formatMetricValue(numberValue, 'percent')}</p>
    </div>
  )
}

function AsrEvalModal({
  model,
  error,
  loading,
  modelOptions,
  modelTypes,
  language,
  onLanguageChange,
  onChange,
  onClose,
  onSubmit,
  onQuickSubmit,
}: {
  model: string
  error?: string
  loading: boolean
  modelOptions: BackendSelectOption[]
  modelTypes?: CapabilitiesResponse['modelTypes']
  language: string
  onLanguageChange: (language: string) => void
  onChange: (model: string) => void
  onClose: () => void
  onSubmit: () => void
  onQuickSubmit: () => void
}) {
  const [informationModel, setInformationModel] = useState<BackendSelectOption | null>(null)
  return createPortal(
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="ASR 测试表单">
      <div className="ui-card max-h-[92vh] w-full max-w-[620px] overflow-y-auto !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-[20px] font-black text-white">ASR 测试</h3>
            <p className="mt-1 text-xs text-slate-500">选择语言和识别模型后开始测试</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭 ASR 测试表单">
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="text-sm font-bold text-slate-300">识别语言</p>
        <div className="mt-2 mb-4 grid grid-cols-2 gap-2">
          {[{ value: 'zh-cn', label: '中文' }, { value: 'en', label: 'English' }].map((item) => (
            <button key={item.value} type="button" onClick={() => onLanguageChange(item.value)} className={cn('h-9 rounded-[7px] border px-3 text-sm font-black', language === item.value ? 'border-cyan-300 bg-cyan-400/14 text-cyan-100' : 'border-cyan-300/12 bg-slate-950/50 text-slate-400')}>
              {item.label}
            </button>
          ))}
        </div>
        <p className="text-sm font-bold text-slate-300">选择 ASR 模型</p>
        <div className="mt-2 grid max-h-[300px] gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
          {modelOptions.map((item) => {
            const unavailable = item.status !== undefined && item.status !== 'available'
            const selected = model === item.value
            return (
              <div key={item.value} className={cn('relative flex min-h-12 items-center rounded-[7px] border px-2 py-2', selected ? 'border-cyan-300 bg-cyan-400/12' : 'border-cyan-300/14 bg-slate-950/55', unavailable && 'opacity-65')}>
                <button type="button" disabled={unavailable} onClick={() => onChange(item.value)} className={cn('min-w-0 flex-1 px-8 text-center text-sm font-bold', selected ? 'text-cyan-100' : 'text-slate-300', unavailable && 'cursor-not-allowed')}>
                  <span className="block truncate">{item.label}</span>
                  {unavailable ? <span className="mt-0.5 block truncate text-[10px] font-medium text-amber-200">暂不可用</span> : null}
                </button>
                <div className="absolute right-2 top-1/2 -translate-y-1/2"><ModelInfoButton model={item} onOpen={setInformationModel} /></div>
              </div>
            )
          })}
        </div>
        {error ? <p className="mt-4 rounded-[7px] border border-red-300/20 bg-red-400/10 px-3 py-2 text-sm text-red-100">{error}</p> : null}
        <div className="mt-5 flex justify-end gap-3">
          <button type="button" onClick={onClose} className="h-10 rounded-[7px] border border-cyan-300/14 bg-white/[0.035] px-4 text-sm font-bold text-slate-300">
            取消
          </button>
          <button type="button" onClick={onQuickSubmit} disabled={loading} className="inline-flex h-10 min-w-[116px] items-center justify-center gap-2 rounded-[7px] border border-cyan-300/22 bg-cyan-400/8 px-4 text-sm font-black text-cyan-100 hover:bg-cyan-400/14 disabled:opacity-60">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            一键测试
          </button>
          <button type="button" onClick={onSubmit} disabled={loading} className="cyan-button inline-flex h-10 min-w-[116px] items-center justify-center gap-2 rounded-[7px] px-4 text-sm font-black disabled:opacity-60">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <TestTube2 className="h-4 w-4" />}
            开始测试
          </button>
        </div>
      </div>
      <ModelInformationModal model={informationModel} modelTypes={modelTypes} onClose={() => setInformationModel(null)} />
    </div>,
    document.body,
  )
}

function annotationSourceLabel(source?: CloneVoiceRequest['annotationSource'] | null) {
  if (source === 'asr') return 'ASR 标注'
  if (source === 'manual') return '人工标注'
  return '—'
}

function cloneAnnotationTitle(
  request: CloneVoiceRequest | undefined,
  modelOption: BackendSelectOption | null,
  fineTune: CloneVoiceResult['fineTune'],
) {
  const explicitSource = request?.annotationSource
  const modelLabel = shortCloneModelName(request?.model ?? modelOption?.value ?? modelOption?.backendValue)
  if (modelLabel === 'XTTS-v2' || modelLabel === 'YourTTS') return null
  if (explicitSource === 'manual') return '人工标注'
  if (explicitSource === 'asr') return 'ASR 标注'
  const asrEvidence = Boolean(
    request?.annotationAsrSubId
      || request?.annotationAsrModel
      || (request?.originalSpeakerPrompt?.trim() && request?.protectedSpeakerPrompt?.trim()),
  )
  const needsAnnotation = cloneModelRequiresReferenceText(modelOption) || asrEvidence || Boolean(fineTune)
  if (!needsAnnotation) return null
  if (asrEvidence) return 'ASR 标注'
  return '人工标注'
}

function cloneModelOption(modelOptions: BackendSelectOption[], model?: string) {
  return modelOptions.find((option) => option.value === model || option.backendValue === model)
}

function shortCloneModelName(value?: string) {
  const model = String(value ?? '')
  if (/xtts[_:/-]?v?2/i.test(model)) return 'XTTS-v2'
  if (/xtts[_:/-]?v?1[._-]?1/i.test(model)) return 'XTTS v1.1'
  if (/your[\s_-]?tts/i.test(model)) return 'YourTTS'
  if (/cosyvoice/i.test(model)) return 'CosyVoice2-0.5B'
  if (/gpt.?sovits/i.test(model)) return 'GPT-SoVITS'
  return model.split('/').at(-1)?.replaceAll('_', ' ') || '—'
}

function shortAsrModelName(value?: string) {
  const model = String(value ?? '')
  const whisper = model.match(/whisper[:/_-]?([a-z0-9.-]+)/i)
  if (whisper?.[1]) return `Whisper ${whisper[1][0].toUpperCase()}${whisper[1].slice(1)}`
  if (/wav2vec/i.test(model)) return 'Wav2Vec2 Base'
  if (/funasr|paraformer/i.test(model)) return 'Paraformer 中文'
  return model.split('/').at(-1)?.replaceAll('_', ' ') || '未生成'
}

function cloneTypeLabel(value?: string) {
  if (/gpt.?sovits/i.test(String(value ?? ''))) return '微调'
  if (/cosyvoice/i.test(String(value ?? ''))) return 'LLM'
  return '零样本'
}

function ModelInfoButton({ model, onOpen }: { model: BackendSelectOption; onOpen: (model: BackendSelectOption) => void }) {
  return (
    <button
      type="button"
      onClick={(event) => {
        event.preventDefault()
        event.stopPropagation()
        onOpen(model)
      }}
      className="grid h-7 w-7 shrink-0 place-items-center rounded-[6px] border border-cyan-300/16 text-cyan-200 transition hover:border-cyan-300/32 hover:bg-cyan-300/10"
      aria-label={`查看 ${model.label} 模型详情`}
      title="查看模型详情"
    >
      <Search className="h-3.5 w-3.5" />
    </button>
  )
}

function AsrHistoryPanel({ history, batches, selectedAsrSubId, onSelect, onOpenBatch, maxVisible, activeLabel }: { history: AsrHistoryEntry[]; batches: EvaluationBatch[]; selectedAsrSubId?: string; onSelect: (asrSubId?: string) => void; onOpenBatch: (batch: EvaluationBatch) => void; maxVisible: number; activeLabel?: string }) {
  if (!history.length && !batches.length) return null
  const rowCount = history.length + batches.length
  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex flex-col items-center justify-center gap-1 text-center">
        <SectionTitle>同一保护任务的 ASR 任务对比</SectionTitle>
        {activeLabel ? <p className="text-sm font-black text-cyan-100">{activeLabel}</p> : null}
        <span className="text-xs text-slate-500">{batches.length ? `${batches.length} 个一键批次 · ` : ''}{history.length} 个模型子任务；点击批次查看最慢任务进度</span>
      </div>
      <div className="mt-4 overflow-auto" style={{ maxHeight: `${42 + Math.max(1, Math.min(rowCount, maxVisible + batches.length)) * 63}px` }}>
        <table className="w-full min-w-[900px] table-fixed text-left text-xs text-slate-300">
          <colgroup>
            <col className="w-[18%]" />
            <col className="w-[23%]" />
            <col className="w-[9%]" />
            <col className="w-[20%]" />
            <col className="w-[12%]" />
            <col className="w-[9%]" />
            <col className="w-[9%]" />
          </colgroup>
          <thead className="sticky top-0 z-10 border-b border-cyan-300/12 bg-slate-950 text-[11px] text-slate-500">
            <tr><th className="px-2 py-2 text-center font-bold text-[13.2px]">任务</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">ASR 模型</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">语言</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">进度</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">处理时长</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">WER</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">CER</th></tr>
          </thead>
          <tbody>
            {batches.map((batch, batchIndex) => {
              const progress = clamp(optionalNumber(batch.progress) ?? 0, 0, 1)
              const progressPercent = Math.round(progress * 100)
              const tone = progressTone(batch.status)
              const elapsed = batchElapsed(batch)
              return (
                <tr key={batch.batchId} role="button" tabIndex={0} onClick={() => onOpenBatch(batch)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onOpenBatch(batch) } }} className="cursor-pointer border-b border-violet-300/14 bg-violet-400/[0.06] hover:bg-violet-400/[0.11] focus:outline-none focus:ring-1 focus:ring-inset focus:ring-violet-300/40">
                  <td className="px-2 py-2 text-center font-mono text-violet-200"><span className="block truncate">一键批次 #{batchIndex + 1}</span><span className="mt-1 block text-[10px] text-slate-500">{formatTaskTime(batch.createdAt)}</span></td>
                  <td className="px-2 py-3 text-center font-black text-violet-100">全模型一键测试</td>
                  <td className="px-2 py-3 text-center">全部</td>
                  <td className="px-2 py-2" title="整体进度以当前最慢的测试为准">
                    <div className="history-progress-track mx-auto h-1.5 max-w-[110px] overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
                    <p className={cn('mt-1 text-center font-mono text-[10px] font-bold', tone.text)}>{progressPercent}% · {lifecycleStatusLabel(batch.status)}</p>
                  </td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">{elapsed !== null ? seconds(elapsed) : '—'}</td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">—</td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">—</td>
                </tr>
              )
            })}
            {history.map((item, index) => {
              const evaluation = item.asr
              const rowId = item.asrSubId ?? `legacy-asr-${index}`
              const lifecycleStatus = asrHistoryLifecycleStatus(item)
              const failed = ['unavailable', 'failed', 'error', 'cancelled'].includes(lifecycleStatus)
              const failureReason = failed ? asrHistoryFailureReason(item) : null
              const progressPercent = Math.round(asrHistoryProgress(item) * 100)
              const statusLabel = asrHistoryStatusLabel(item)
              const elapsedSec = optionalNumber(item.taskStatus?.elapsedSec)
              const tone = progressTone(lifecycleStatus)
              return (
                <tr key={rowId} role="button" tabIndex={0} title={failureReason ?? item.taskStatus?.message ?? undefined} onClick={() => onSelect(item.asrSubId)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(item.asrSubId) } }} className={cn('cursor-pointer border-b border-cyan-300/8 last:border-0 hover:bg-cyan-300/[0.04] focus:outline-none focus:ring-1 focus:ring-inset focus:ring-cyan-300/35', selectedAsrSubId === item.asrSubId && 'bg-cyan-300/[0.08]', failed && 'bg-rose-400/[0.035]')}>
                  <td className="px-2 py-2 text-center font-mono text-cyan-200"><span className="block truncate">ASR 测试 #{index + 1}{failed ? <span className="ml-2 rounded-full bg-rose-400/12 px-1.5 py-0.5 font-sans text-[10px] font-black text-rose-300">失败</span> : null}</span><span className="mt-1 block text-[10px] text-slate-500">{formatTaskTime(item.taskStatus?.createdAt ?? item.createdAt ?? evaluation?.createdAt)}</span></td>
                  <td className={cn('truncate px-2 py-3 text-center font-bold', failed ? 'text-rose-200' : 'text-slate-100')} title={evaluation?.model ?? item.request?.model}>{shortAsrModelName(evaluation?.model ?? item.request?.model)}</td>
                  <td className="px-2 py-3 text-center">{evaluation?.language ?? item.request?.language ?? '—'}</td>
                  <td className="px-2 py-2" title={item.taskStatus?.message ?? statusLabel}>
                    <div className="history-progress-track mx-auto h-1.5 max-w-[110px] overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
                    <p className={cn('mt-1 text-center font-mono text-[10px] font-bold', tone.text)}>{progressPercent}% · {statusLabel}</p>
                  </td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">{elapsedSec !== null ? seconds(elapsedSec) : '—'}</td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">{formatAsrRatePercent(evaluation?.wer)}</td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">{formatAsrRatePercent(evaluation?.cer)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function CloneHistoryPanel({ history, batches, modelOptions, selectedCloneKey, onSelect, onOpenBatch, onOpenAsr, onOpenManual, maxVisible, activeLabel, activeModel, activeModelOption, onOpenModel }: { history: CloneHistoryEntry[]; batches: EvaluationBatch[]; modelOptions: BackendSelectOption[]; selectedCloneKey?: string; onSelect: (cloneKey: string) => void; onOpenBatch: (batch: EvaluationBatch) => void; onOpenAsr: (asrSubId?: string) => void; onOpenManual: (item: CloneHistoryEntry) => void; maxVisible: number; activeLabel?: string; activeModel?: string; activeModelOption?: BackendSelectOption | null; onOpenModel: (model: BackendSelectOption) => void }) {
  if (!history.length && !batches.length) return null
  const rowCount = history.length + batches.length
  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex flex-col items-center justify-center gap-1 text-center">
        <SectionTitle>同一保护任务的克隆任务对比</SectionTitle>
        {activeModel ? (
          <div className="flex items-center justify-center gap-2 text-sm font-black text-violet-100">
            <span>TTS 克隆{activeLabel ? ` · ${activeLabel}` : ''} · {activeModel}</span>
            {activeModelOption ? <ModelInfoButton model={activeModelOption} onOpen={onOpenModel} /> : null}
          </div>
        ) : null}
        <span className="text-xs text-slate-500">{batches.length ? `${batches.length} 个一键批次 · ` : ''}{history.length} 个克隆子任务；排队、运行、失败记录均会保留</span>
      </div>
      <div className="mt-4 overflow-auto" style={{ maxHeight: `${42 + Math.max(1, Math.min(rowCount, maxVisible + batches.length)) * 63}px` }}>
        <table className="w-full min-w-[1280px] table-fixed text-left text-xs text-slate-300">
          <colgroup>
            <col className="w-[12%]" />
            <col className="w-[8%]" />
            <col className="w-[13%]" />
            <col className="w-[9%]" />
            <col className="w-[15%]" />
            <col className="w-[8%]" />
            <col className="w-[21%]" />
            <col className="w-[7%]" />
            <col className="w-[7%]" />
          </colgroup>
          <thead className="sticky top-0 z-10 border-b border-cyan-300/12 bg-slate-950 text-[11px] text-slate-500">
            <tr><th className="px-2 py-2 text-center font-bold text-[13.2px]">任务</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">克隆类型</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">克隆模型</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">标注来源</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">进度</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">处理时长</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">参考标注</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">原始相似度</th><th className="px-2 py-2 text-center font-bold text-[13.2px]">保护后相似度</th></tr>
          </thead>
          <tbody>
            {batches.map((batch, batchIndex) => {
              const progressPercent = Math.round(clamp(optionalNumber(batch.progress) ?? 0, 0, 1) * 100)
              const tone = progressTone(batch.status)
              const elapsed = batchElapsed(batch)
              return (
                <tr key={batch.batchId} role="button" tabIndex={0} onClick={() => onOpenBatch(batch)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onOpenBatch(batch) } }} className="cursor-pointer border-b border-violet-300/14 bg-violet-400/[0.06] hover:bg-violet-400/[0.11] focus:outline-none focus:ring-1 focus:ring-inset focus:ring-violet-300/40">
                  <td className="px-2 py-2 text-center font-mono text-violet-200"><span className="block truncate">一键批次 #{batchIndex + 1}</span><span className="mt-1 block text-[10px] text-slate-500">{formatTaskTime(batch.createdAt)}</span></td>
                  <td className="px-2 py-3 text-center"><span className="rounded-full border border-violet-300/20 bg-violet-400/10 px-2 py-1 font-black text-violet-100">批次</span></td>
                  <td className="px-2 py-3 text-center font-black text-violet-100">全模型一键测试</td>
                  <td className="px-2 py-3 text-center">—</td>
                  <td className="px-2 py-2" title="整体进度以当前最慢的测试为准">
                    <div className="history-progress-track mx-auto h-1.5 max-w-[110px] overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
                    <p className={cn('mt-1 text-center font-mono text-[10px] font-bold', tone.text)}>{progressPercent}% · {lifecycleStatusLabel(batch.status)}</p>
                  </td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">{elapsed !== null ? seconds(elapsed) : '—'}</td>
                  <td className="px-2 py-3 text-center text-slate-500">—</td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">—</td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">—</td>
                </tr>
              )
            })}
            {history.map((item, index) => {
              const request = item.request
              const requiresReferenceText = cloneModelRequiresReferenceText(cloneModelOption(modelOptions, request?.model))
              const annotationSource = requiresReferenceText ? request?.annotationSource : undefined
              const hasAnnotation = annotationSource === 'manual' || annotationSource === 'asr'
              const asrAnnotation = annotationSource === 'asr'
              const originalPrompt = request?.originalSpeakerPrompt ?? request?.speakerPrompt ?? ''
              const protectedPrompt = request?.protectedSpeakerPrompt ?? ''
              const lifecycleStatus = cloneHistoryLifecycleStatus(item)
              const failed = ['unavailable', 'failed', 'error', 'cancelled'].includes(lifecycleStatus)
              const progressPercent = Math.round(cloneHistoryProgress(item) * 100)
              const tone = progressTone(lifecycleStatus)
              const elapsed = optionalNumber(item.taskStatus?.elapsedSec)
              const openAnnotation = () => {
                if (!request || !hasAnnotation) return
                if (asrAnnotation) onOpenAsr(request.annotationAsrSubId)
                else onOpenManual(item)
              }
              return (
                <tr key={item.key} role="button" tabIndex={0} title={failed ? cloneHistoryFailureReason(item) : item.taskStatus?.message ?? undefined} onClick={() => onSelect(item.key)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(item.key) } }} className={cn('cursor-pointer border-b border-cyan-300/8 last:border-0 hover:bg-cyan-300/[0.04] focus:outline-none focus:ring-1 focus:ring-inset focus:ring-cyan-300/35', selectedCloneKey === item.key && 'bg-cyan-300/[0.08]', failed && 'bg-rose-400/[0.035]')}>
                  <td className="px-2 py-2 text-center font-mono text-cyan-200"><span className="block truncate">克隆测试 #{index + 1}{failed ? <span className="ml-2 rounded-full bg-rose-400/12 px-1.5 py-0.5 font-sans text-[10px] font-black text-rose-300">失败</span> : null}</span><span className="mt-1 block text-[10px] text-slate-500">{formatTaskTime(item.taskStatus?.createdAt ?? item.createdAt)}</span></td>
                  <td className="px-2 py-3 text-center">
                    <span className="rounded-full border border-cyan-300/18 bg-cyan-400/8 px-2 py-1 font-black text-cyan-100">{request?.model ? cloneTypeLabel(request.model) : '—'}</span>
                  </td>
                  <td className={cn('truncate px-2 py-3 text-center font-bold', failed ? 'text-rose-200' : 'text-slate-100')} title={request?.model}>{shortCloneModelName(request?.model)}</td>
                  <td className="px-2 py-3 text-center">
                    {hasAnnotation ? <button type="button" onClick={(event) => { event.stopPropagation(); openAnnotation() }} className={cn('rounded-full border px-2 py-1 font-bold underline-offset-2 hover:underline', asrAnnotation ? 'border-violet-300/20 bg-violet-400/10 text-violet-200' : 'manual-annotation-chip border-emerald-400/30 bg-emerald-500/15 text-emerald-300')}>{annotationSourceLabel(annotationSource)}</button> : '—'}
                  </td>
                  <td className="px-2 py-2" title={item.taskStatus?.message ?? lifecycleStatusLabel(lifecycleStatus)}>
                    <div className="history-progress-track mx-auto h-1.5 max-w-[110px] overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
                    <p className={cn('mt-1 text-center font-mono text-[10px] font-bold', tone.text)}>{progressPercent}% · {lifecycleStatusLabel(lifecycleStatus)}</p>
                  </td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">{elapsed !== null ? seconds(elapsed) : '—'}</td>
                  <td className="px-2 py-2">
                    {hasAnnotation ? <button type="button" onClick={(event) => { event.stopPropagation(); openAnnotation() }} className="block w-full rounded-[5px] px-1 py-1 text-left hover:bg-cyan-300/[0.05]">{asrAnnotation ? <span className="block space-y-0.5 leading-5"><span className="block truncate" title={originalPrompt}>原始：{originalPrompt || '—'}</span><span className="block truncate" title={protectedPrompt}>保护：{protectedPrompt || '—'}</span></span> : <span className="block truncate" title={originalPrompt}>{originalPrompt || '—'}</span>}</button> : <span className="block text-center text-slate-500">—</span>}
                  </td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">{formatCloneMetricNumber(item.result?.cloneEval?.originalSimilarity)}</td>
                  <td className="px-2 py-3 text-center font-mono font-bold text-[13.2px]">{formatCloneMetricNumber(item.result?.cloneEval?.protectedSimilarity)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function ManualAnnotationModal({ item, onClose }: { item: CloneHistoryEntry | null; onClose: () => void }) {
  useEffect(() => {
    if (!item) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [item, onClose])
  if (!item) return null
  const request = item.request
  if (!request) return null
  const text = request.speakerPrompt ?? request.originalSpeakerPrompt ?? ''
  return createPortal(
    <div className="fixed inset-0 z-[110] grid place-items-center bg-slate-950/72 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="人工标注详情" onClick={onClose}>
      <div className="ui-card w-full max-w-[620px] !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.48)]" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-black text-white">人工标注详情</h3>
            <p className="mt-1 text-xs text-slate-500">TTS 克隆 · {shortCloneModelName(request.model)}</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-cyan-300/14 text-slate-300 hover:text-white" aria-label="关闭人工标注详情"><X className="h-4 w-4" /></button>
        </div>
        <div className="mt-4 rounded-[8px] border border-emerald-300/16 bg-emerald-400/[0.06] p-4">
          <p className="text-xs font-black text-emerald-300">原始与保护参考音频共用这一条人工标注</p>
          <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-200">{text || '未填写人工标注'}</p>
        </div>
      </div>
    </div>,
    document.body,
  )
}

function CloneVoiceModal({
  form,
  error,
  loading,
  modelOptions,
  modelTypes,
  languageOptions,
  speedOptions,
  asrAnnotations,
  onChange,
  onClose,
  onSubmit,
  onQuickSubmit,
  onOpenAsr,
}: {
  form: CloneVoiceRequest
  error?: string
  loading: boolean
  modelOptions: BackendSelectOption[]
  modelTypes?: CapabilitiesResponse['modelTypes']
  languageOptions: string[]
  speedOptions: number[]
  asrAnnotations: AsrEvalResponse[]
  onChange: (form: CloneVoiceRequest) => void
  onClose: () => void
  onSubmit: () => void
  onQuickSubmit: () => void
  onOpenAsr: () => void
}) {
  const [annotationSearch, setAnnotationSearch] = useState('')
  const [informationModel, setInformationModel] = useState<BackendSelectOption | null>(null)
  const visibleModels = modelOptions
  const selectedModel = cloneModelOption(modelOptions, form.model)
  const reusableAsrAnnotations = asrAnnotations
    .filter((item) => item.asrSubId && item.asr?.originalText?.trim() && item.asr?.protectedText?.trim())
    .filter((item) => {
      const query = annotationSearch.trim().toLowerCase()
      if (!query) return true
      return `${item.asr?.originalText ?? ''} ${item.asr?.protectedText ?? ''} ${item.asr?.model ?? ''}`.toLowerCase().includes(query)
    })
    .slice()
    .reverse()

  const selectModel = (selected: BackendSelectOption) => {
    const languages = selected.languages?.length ? selected.languages : languageOptions
    onChange(normalizeCloneReferenceTextRequest({
      ...form,
      model: selected.value,
      language: languages.includes(form.language ?? '') ? form.language : languages[0],
    }, selected))
  }

  return createPortal(
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="语音克隆测试表单">
      <div className="ui-card max-h-[92vh] w-full max-w-[620px] overflow-y-auto !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-[20px] font-black text-white">语音克隆测试</h3>
            <p className="mt-1 text-xs text-slate-500">选择文本和模型{cloneModelRequiresReferenceText(selectedModel) ? '，并提供参考标注' : ''}后开始测试</p>
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
        <p className="mt-4 text-sm font-bold text-slate-300">模型</p>
        <div className="mt-2 grid max-h-[176px] gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
          {visibleModels.map((item) => {
            const unavailable = item.status !== undefined && item.status !== 'available'
            const selected = form.model === item.value
            return (
              <div key={item.value} className={cn('relative flex min-h-12 items-center rounded-[7px] border px-2 py-2', selected ? 'border-cyan-300 bg-cyan-400/12' : 'border-cyan-300/14 bg-slate-950/55', unavailable && 'opacity-65')}>
                <button type="button" disabled={unavailable} onClick={() => selectModel(item)} className={cn('min-w-0 flex-1 px-8 text-center text-sm font-bold', selected ? 'text-cyan-100' : 'text-slate-300', unavailable && 'cursor-not-allowed')}>
                  <span className="block truncate">{item.label}</span>
                  {unavailable ? <span className="mt-0.5 block truncate text-[10px] font-medium text-amber-200">暂不可用</span> : null}
                </button>
                <div className="absolute right-2 top-1/2 -translate-y-1/2"><ModelInfoButton model={item} onOpen={setInformationModel} /></div>
              </div>
            )
          })}
        </div>
        <div className="mt-4 grid grid-cols-[1fr_120px] gap-3">
          <label className="text-sm font-bold text-slate-300">
            语言
            <select value={form.language ?? 'auto'} onChange={(event) => onChange({ ...form, language: event.target.value })} className="mt-2 h-10 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950 px-3 text-slate-100 outline-none focus:border-cyan-300">
              {languageOptions.map((item) => (
                <option key={item} value={item} className="bg-slate-950 text-slate-100">
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm font-bold text-slate-300">
            语速
            <select value={String(form.speed ?? 1)} onChange={(event) => onChange({ ...form, speed: Number(event.target.value) })} className="mt-2 h-10 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950 px-3 text-slate-100 outline-none focus:border-cyan-300">
              {speedOptions.map((item) => (
                <option key={item} value={item} className="bg-slate-950 text-slate-100">
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>
        {cloneModelRequiresReferenceText(selectedModel) ? (
          <div className="mt-4 rounded-[7px] border border-cyan-300/14 bg-slate-950/35 p-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm font-bold text-slate-300">参考音频标注（必填）</p>
              <div className="flex gap-2">
                {(['manual', 'asr'] as const).map((source) => (
                  <button key={source} type="button" onClick={() => onChange({ ...form, annotationSource: source, annotationAsrSubId: source === 'manual' ? undefined : form.annotationAsrSubId, annotationAsrModel: source === 'manual' ? undefined : form.annotationAsrModel, originalSpeakerPrompt: source === 'manual' ? undefined : form.originalSpeakerPrompt, protectedSpeakerPrompt: source === 'manual' ? undefined : form.protectedSpeakerPrompt })} className={cn('h-8 rounded-[6px] border px-3 text-xs font-black', (form.annotationSource ?? 'manual') === source ? 'border-cyan-300 bg-cyan-400/14 text-cyan-100' : 'border-cyan-300/12 text-slate-400')}>{annotationSourceLabel(source)}</button>
                ))}
              </div>
            </div>
            <div className="mt-3 rounded-[6px] border border-cyan-300/10 bg-cyan-300/[0.035] px-3 py-2 text-[11px] leading-5 text-slate-400">
              {(form.annotationSource ?? 'manual') === 'asr' ? (
                <><p>• ASR 标注同时包含原始音频转写与保护音频转写。</p><p>• 克隆原语音使用原始转写。</p><p>• 克隆保护语音使用保护转写。</p></>
              ) : (
                <><p>• 人工标注只有一条文本。</p><p>• 原始参考音频与保护参考音频共同使用该文本。</p></>
              )}
            </div>
            {(form.annotationSource ?? 'manual') === 'manual' ? (
              <input value={form.speakerPrompt ?? ''} onChange={(event) => onChange({ ...form, speakerPrompt: event.target.value, originalSpeakerPrompt: undefined, protectedSpeakerPrompt: undefined, annotationSource: 'manual', annotationAsrSubId: undefined, annotationAsrModel: undefined, annotationCreatedAt: undefined })} className="mt-3 h-10 w-full rounded-[7px] border border-cyan-300/14 bg-slate-950/70 px-3 text-slate-100 outline-none focus:border-cyan-300" placeholder="输入一条人工核对后的参考音频文本" />
            ) : reusableAsrAnnotations.length || annotationSearch ? (
              <div className="mt-3">
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-slate-500" />
                  <input value={annotationSearch} onChange={(event) => setAnnotationSearch(event.target.value)} className="h-10 w-full rounded-[7px] border border-violet-300/14 bg-slate-950/70 pl-9 pr-3 text-slate-100 outline-none focus:border-violet-300" placeholder="搜索最近 ASR 文本、模型或保护任务" />
                </div>
                <div className="mt-2 max-h-[150px] space-y-2 overflow-y-auto pr-1">
                  {reusableAsrAnnotations.map((item) => (
                    <button key={item.asrSubId} type="button" onClick={() => onChange({ ...form, annotationSource: 'asr', annotationAsrSubId: item.asrSubId, annotationAsrModel: item.asr?.model, annotationCreatedAt: item.createdAt ?? undefined, speakerPrompt: item.asr?.originalText ?? '', originalSpeakerPrompt: item.asr?.originalText ?? '', protectedSpeakerPrompt: item.asr?.protectedText ?? '' })} className={cn('w-full rounded-[7px] border p-3 text-left', form.annotationAsrSubId === item.asrSubId ? 'border-violet-300 bg-violet-400/12' : 'border-violet-300/12 bg-slate-950/50 hover:bg-violet-400/[0.06]')}>
                      <p className="truncate text-xs font-black text-violet-100">{shortAsrModelName(item.asr?.model)} · {item.createdAt ? formatTaskTime(item.createdAt) : '最近结果'}</p>
                      <p className="mt-1 truncate text-xs leading-5 text-slate-300">原始：{item.asr?.originalText}</p>
                      <p className="truncate text-xs leading-5 text-slate-400">保护：{item.asr?.protectedText}</p>
                    </button>
                  ))}
                  {!reusableAsrAnnotations.length ? <p className="py-3 text-center text-xs text-slate-500">没有匹配的 ASR 标注</p> : null}
                </div>
              </div>
            ) : (
              <button type="button" onClick={onOpenAsr} className="mt-3 w-full rounded-[7px] border border-dashed border-violet-300/25 bg-violet-400/[0.06] px-3 py-4 text-sm font-black text-violet-100 hover:bg-violet-400/10">当前没有 ASR 标注，先运行 ASR 测试</button>
            )}
          </div>
        ) : null}
        {error ? <p className="mt-4 rounded-[7px] border border-red-300/20 bg-red-400/10 px-3 py-2 text-sm text-red-100">{error}</p> : null}
        <div className="mt-5 flex justify-end gap-3">
          <button type="button" onClick={onClose} className="h-10 rounded-[7px] border border-cyan-300/14 bg-white/[0.035] px-4 text-sm font-bold text-slate-300">
            取消
          </button>
          <button type="button" onClick={onQuickSubmit} disabled={loading} className="inline-flex h-10 min-w-[128px] items-center justify-center gap-2 rounded-[7px] border border-cyan-300/22 bg-cyan-400/8 px-4 text-sm font-black text-cyan-100 hover:bg-cyan-400/14 disabled:opacity-60">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            一键克隆
          </button>
          <button type="button" onClick={onSubmit} disabled={loading} className="cyan-button inline-flex h-10 min-w-[128px] items-center justify-center gap-2 rounded-[7px] px-4 text-sm font-black disabled:opacity-60">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <TestTube2 className="h-4 w-4" />}
            开始测试
          </button>
        </div>
      </div>
      <ModelInformationModal model={informationModel} modelTypes={modelTypes} onClose={() => setInformationModel(null)} />
    </div>,
    document.body,
  )
}

function AudioCard({
  title,
  audio,
  color,
  green,
  onPlayRequest,
  compactHeader = false,
}: {
  title: string
  audio: AudioFileMeta
  color: string
  green?: boolean
  onPlayRequest?: () => Promise<string | undefined>
  compactHeader?: boolean
}) {
  const src = getAudioSource(audio)
  const duration = getAudioDuration(audio)

  return (
    <div className={cn('result-audio-card flex h-[252px] flex-col rounded-[9px] border p-5', green ? 'result-audio-card-protected border-emerald-400/18 bg-emerald-400/8' : 'border-cyan-300/14 bg-[#07192d]/80')}>
      {!compactHeader ? (
        <>
          <p className="flex items-center gap-2 whitespace-nowrap text-sm font-black text-slate-200">
            {green ? <ShieldCheck className="h-4 w-4 text-emerald-300" /> : <Volume2 className="h-4 w-4 text-sky-300" />}
            {title}
          </p>
          <p className="ml-6 mt-0.5 flex min-w-0 text-xs text-slate-400">
            <span className="truncate">{audio.filename.replace(/\.[^.]+$/, '')}</span>
            <span className="shrink-0">{audio.filename.match(/\.[^.]+$/)?.[0] ?? ''}</span>
          </p>
        </>
      ) : null}
      <TinyWave color={color} className="h-[58px]" />
      <div className="mt-auto">
        <AudioPlayer
          src={src}
          title={title}
          filename={audio.filename}
          disabledReason={green ? '点击播放时将在线获取保护音频' : '暂无原始音频 URL'}
          downloadable={Boolean(src)}
          downloadFilename={audio.filename}
          onPlayRequest={onPlayRequest}
        />
      </div>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 border-t border-cyan-300/10 pt-3 pb-1 text-[12px] text-slate-400">
        <span>时长 {duration ? `${duration.toFixed(2)}s` : '待解析'}</span>
        <span>采样率 {audio.sampleRate ? `${(audio.sampleRate / 1000).toFixed(2)}kHz` : '待解析'}</span>
        <span>声道 {audio.channels ?? '待解析'}</span>
        <span>格式 {audio.format}</span>
        <span>大小 {formatFileSize(audio.sizeBytes)}</span>
      </div>
    </div>
  )
}

function TextBox({ title, text, foot, content }: { title: string; text: string; foot: string; content?: ReactNode }) {
  return (
    <div className="flex h-full flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/18 p-4">
      <div className="mb-3 flex items-start gap-2">
        <h3 className="min-w-0 flex-1 whitespace-nowrap text-sm font-bold text-slate-300">{title}</h3>
        <MetricInfoButton title={title}>{foot}</MetricInfoButton>
      </div>
      <div className="mb-3 h-[156px] shrink-0 overflow-y-auto rounded-[7px] border border-cyan-300/8 bg-slate-950/22 px-4 py-3 text-[13px] leading-6 text-slate-200">
        {content ?? text}
      </div>
    </div>
  )
}

function BatchProgressModal({ batch, cloneModelOptions, onClose }: { batch: EvaluationBatch | null; cloneModelOptions: BackendSelectOption[]; onClose: () => void }) {
  useEffect(() => {
    if (!batch) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [batch, onClose])
  if (!batch) return null
  const progress = clamp(optionalNumber(batch.progress) ?? 0, 0, 1)
  const progressPercent = Math.round(progress * 100)
  const tone = progressTone(batch.status)
  const title = batch.type === 'asr' ? '一键 ASR 测试进度' : '一键克隆测试进度'
  return createPortal(
    <div className="fixed inset-0 z-[150] grid place-items-center bg-slate-950/80 px-4 py-8 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label={title} onClick={onClose}>
      <div className="ui-card flex max-h-full w-full max-w-[900px] flex-col overflow-hidden !bg-[#061426] p-5 shadow-[0_28px_90px_rgba(0,0,0,0.62)]" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-xl font-black text-white">{title}</h3>
            <p className="mt-1 text-xs text-slate-500">{batch.type === 'asr' ? '全部 ASR 模型' : '全部克隆模型'} · {batch.totalCount} 个测试</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 text-slate-300 hover:text-white" aria-label="关闭"><X className="h-4 w-4" /></button>
        </div>
        <div className="mt-5 rounded-[9px] border border-violet-300/16 bg-violet-400/[0.07] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
            <span className="font-black text-violet-100">整体 {progressPercent}% · {lifecycleStatusLabel(batch.status)}</span>
            <span className="text-slate-400">完成 {batch.completedCount}/{batch.totalCount} · 失败 {batch.failedCount}</span>
          </div>
          <div className="history-progress-track mt-3 h-2 overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
          <p className="mt-2 text-xs text-slate-500">整体进度以当前最慢的测试为准，便于判断这一批任务还需要等待多久。</p>
        </div>
        <div className="mt-4 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
          {batch.items.map((item) => {
            const itemProgress = clamp(optionalNumber(item.progress) ?? 0, 0, 1)
            const itemPercent = Math.round(itemProgress * 100)
            const itemTone = progressTone(item.status)
            const itemError = typeof item.error === 'string' ? item.error : item.error?.message
            const modelLabel = item.modelName || (batch.type === 'asr' ? shortAsrModelName(item.model) : shortCloneModelName(item.model))
            const showAnnotationSource = batch.type === 'clone' && cloneModelRequiresReferenceText(cloneModelOption(cloneModelOptions, item.model)) && Boolean(item.annotationSource)
            return (
              <div key={item.batchItemId} className={cn('rounded-[8px] border p-3', ['failed', 'error', 'cancelled'].includes(String(item.status).toLowerCase()) ? 'border-rose-300/18 bg-rose-400/[0.05]' : 'border-cyan-300/10 bg-slate-950/24')}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate font-black text-slate-100" title={item.model}>{modelLabel}</p>
                    <p className="mt-0.5 text-[11px] text-slate-500">{item.modelType || (batch.type === 'asr' ? 'ASR' : 'TTS 克隆')}{showAnnotationSource ? ` · ${annotationSourceLabel(item.annotationSource)}` : ''}</p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className={cn('font-mono text-xs font-black', itemTone.text)}>{itemPercent}% · {lifecycleStatusLabel(item.status)}</p>
                    <p className="mt-0.5 font-mono text-[10px] text-slate-500">{optionalNumber(item.elapsedSec) !== null ? seconds(optionalNumber(item.elapsedSec) as number) : '—'}</p>
                  </div>
                </div>
                <div className="history-progress-track mt-2 h-1.5 overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', itemTone.fill)} style={{ width: `${itemPercent}%` }} /></div>
                <p className={cn('mt-2 truncate text-xs', itemError ? 'text-rose-300' : 'text-slate-400')} title={itemError || item.message || undefined}>{itemError || item.message || '等待任务状态更新'}</p>
              </div>
            )
          })}
        </div>
      </div>
    </div>,
    document.body,
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

function renderDiffOps(diffOps: DiffOp[]): ReactNode[] {
  const diffTokens = diffOps.flatMap((op) => ('text' in op ? [op.text] : [op.from, op.to])).filter(Boolean)
  const lexicalTokens = diffTokens.filter((token) => /[\p{L}\p{N}]/u.test(token))
  const latinTokens = lexicalTokens.filter((token) => /[\p{Script=Latin}\p{N}]/u.test(token))
  const wordLevelLatin = latinTokens.length > 0 && latinTokens.length / Math.max(lexicalTokens.length, 1) >= 0.6
  const joiner = wordLevelLatin ? ' ' : ''
  const nodes: ReactNode[] = []
  diffOps.forEach((op, index) => {
    const spacer = index === diffOps.length - 1 ? '' : joiner
    if (op.type === 'equal') {
      nodes.push(`${op.text}${spacer}`)
      return
    }
    if (op.type === 'insert') {
      nodes.push(
        <span key={`ins-${index}`} className="rounded-[3px] bg-red-400/[0.07] px-0.5 text-red-300 transition-colors duration-200 hover:bg-red-400/[0.16]" title="新增内容">
          {op.text}
          {spacer}
        </span>,
      )
      return
    }
    if (op.type === 'delete') {
      nodes.push(
        <span key={`del-${index}`} className="rounded-[3px] bg-emerald-400/[0.06] px-0.5 text-emerald-300 line-through decoration-emerald-300/70 transition-colors duration-200 hover:bg-emerald-400/[0.14]" title="缺失内容">
          {op.text}
          {spacer}
        </span>,
      )
      return
    }
    if ('from' in op) {
      nodes.push(
        <span key={`replace-del-${index}`} className="rounded-[3px] bg-emerald-400/[0.06] px-0.5 text-emerald-300 line-through decoration-emerald-300/70 transition-colors duration-200 hover:bg-emerald-400/[0.14]" title="替换前内容">
          {op.from}
        </span>,
        <span key={`replace-ins-${index}`} className="rounded-[3px] bg-red-400/[0.07] px-0.5 text-red-300 transition-colors duration-200 hover:bg-red-400/[0.16]" title="替换后内容">
          {joiner}
          {op.to}
          {spacer}
        </span>,
      )
    }
  })
  return nodes
}

function ScoreBox({ label, value, red, compact, foot }: { label: ReactNode; value: string; red?: boolean; compact?: boolean; foot?: ReactNode }) {
  return (
    <MetricInfoSurface title={label} info={foot} className={cn('relative rounded-[9px] border border-cyan-300/12 bg-slate-950/16 text-center', compact ? 'p-2.5' : 'p-3')}>
      <div className={cn('relative', compact ? 'min-h-9' : 'min-h-8')}>
        <p className={cn('absolute inset-x-0 top-1/2 line-clamp-2 -translate-y-1/2 break-words px-2 text-center font-black text-slate-300', compact ? 'text-[12px] leading-4' : 'text-[13px] leading-5')}>{label}</p>
      </div>
      <div className="mt-2 grid justify-items-center">
        <span className={cn(compact ? 'text-[19px]' : 'text-[24px]', 'break-words font-black leading-none', red ? 'text-red-300' : 'text-cyan-300')}>
          {value}
        </span>
      </div>
    </MetricInfoSurface>
  )
}

function QualityPanel({ result, embedded }: { result: TaskResult; embedded?: boolean }) {
  const snr = optionalNumber(result.protectionQuality?.snr) ?? optionalNumber(result.quality.snr)
  const pesq = optionalNumber(result.protectionQuality?.pesq) ?? optionalNumber(result.quality.pesq)
  const stoi = optionalNumber(result.protectionQuality?.stoi)
  const dnsMos = optionalNumber(result.protectionQuality?.dnsMos)
  const missingReasons = [
    pesq === null ? ['PESQ', metricReason(result, ['protectionQuality.pesq'])] : null,
    stoi === null ? ['STOI', metricReason(result, ['protectionQuality.stoi'])] : null,
    dnsMos === null ? ['DNSMOS', result.protectionQuality?.dnsMosReason ? friendlyCloneMetricReason(result.protectionQuality.dnsMosReason) : metricReason(result, ['protectionQuality.dnsMos'])] : null,
  ].filter((item): item is [string, string] => Boolean(item?.[1]))

  return (
    <section className={cn(embedded ? 'mt-5' : 'ui-card p-5')}>
      <SectionTitle>感知质量评估</SectionTitle>
      <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(132px,1fr))] gap-3">
        <QualityMetric
          label="SNR"
          value={formatMetricValue(snr, 'db')}
          tag={snr === null ? '未生成' : '信噪比'}
          tone="green"
          foot={(
            <MetricFormulaContent
              description="SNR 是 Signal-to-Noise Ratio（信噪比）。在本系统中，有效信号是原始音频，噪声是保护音频减去原始音频得到的保护扰动，用于衡量原音功率与扰动功率的关系。"
              formulas={[
                "\\delta=x^{\\prime}-x,\\qquad P_x=\\operatorname{mean}(x^2),\\qquad P_{\\delta}=\\operatorname{mean}(\\delta^2)",
                '\\mathrm{SNR}=10\\log_{10}\\!\\left(\\frac{P_x+10^{-12}}{P_{\\delta}+10^{-12}}\\right)\\,\\mathrm{dB}',
                '\\mathrm{SNR}\\in(-\\infty,+\\infty)\\,\\mathrm{dB}',
              ]}
              note={<>SNR 没有固定上限，数值越高表示保护音频越接近原音、扰动越不明显；低值表示扰动功率相对更高、听感可能更容易受影响，但不等同于保护一定更有效。<MathText formula="10,20,30\\,\\mathrm{dB}" className="mx-0.5 align-[-1px]" /> 分别约对应原音功率为扰动功率的 <MathText formula="10,100,1000" className="mx-0.5 align-[-1px]" /> 倍；在防护任务中不能脱离身份与语义保护效果而单独追求更高 SNR。</>}
            />
          )}
        />
        <QualityMetric
          label="PESQ"
          value={formatMetricValue(pesq, 'number')}
          tag={pesq === null ? '未生成' : '听感质量'}
          tone="blue"
          foot={(
            <MetricFormulaContent
              description="PESQ 是 Perceptual Evaluation of Speech Quality（语音质量感知评估），是一项有参考指标，通过比较原始参考语音与保护语音，综合评估噪声、失真、编码和时序偏差对听感质量的影响。"
              formulas={['\\text{PESQ 常见范围约为 }-0.5\\text{～}4.5\\text{，无量纲}']}
              note="PESQ 没有物理单位，分数越高越好。声音发闷、明显失真或带有机械伪影时，分数通常会下降；原始语音本身的“呃”等语气或内容不是 PESQ 专门判断的对象，只有处理新增的停顿、尾音延长等变化偏离参考语音时，才可能间接导致分数下降。"
            />
          )}
        />
        <QualityMetric
          label="STOI"
          value={formatMetricValue(stoi, 'number')}
          tag={stoi === null ? '未生成' : '可懂度'}
          tone="blue"
          foot={(
            <MetricFormulaContent
              description="STOI 是 Short-Time Objective Intelligibility（短时客观可懂度），是一项有参考指标，通过比较短时频带包络，衡量元音、辅音和音节是否仍能被清楚辨认。"
              formulas={['\\mathrm{STOI}\\in[0,1]\\text{，无量纲}']}
              note="STOI 没有物理单位，越接近 1 越好。例如句子每个音节都清楚时得分较高；噪声掩盖辅音、导致词语难以分辨时得分会降低。STOI 衡量可懂度，不等同于自然度，也不直接判断文本语义。"
            />
          )}
        />
        <QualityMetric
          label={<span className="block min-w-0 whitespace-nowrap text-[11px] sm:text-[12px]"><span className="hidden sm:inline">DNSMOS</span><span className="sm:hidden">DNS</span></span>}
          value={formatMetricValue(dnsMos, 'number')}
          tag={dnsMos === null ? '未生成' : '语音质量评分'}
          tone="orange"
          foot={(
            <MetricFormulaContent
              description="DNSMOS 是 Deep Noise Suppression Mean Opinion Score（深度降噪平均意见分），是一种不需要参考音频的语音质量估计方法。P.835 同时给出 SIG（语音信号质量）、BAK（背景噪声质量）和 OVRL（总体质量）；这里展示保护音频的 OVRL。"
              formulas={['\\mathrm{DNSMOS}_{\\mathrm{OVRL}}\\in[1,5]\\text{ 分}']}
              note="单位为分，数值越高越好。OVRL 综合反映语音失真、背景噪声和自然度，例如清晰自然、背景干净的语音通常得分较高，噪声或失真明显时得分较低。该结果是模型预测，不是人工现场评分，也不应作为绝对听感结论。"
            />
          )}
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
  const modeTime = (actualTimeSec ?? selectedTimeSec ?? 0).toFixed(2)
  const modeLabel = psychoMode === 'frame'
    ? <><MathText formula={`t=${modeTime}\\,\\mathrm{s}`} className="align-[-1px]" /> 对应帧</>
    : <><MathText formula="t" className="align-[-1px]" /> 平均聚合</>
  const modeDescription =
    psychoMode === 'frame'
      ? frameIndex !== null && actualTimeSec !== null
        ? `当前显示 \\(t=${actualTimeSec.toFixed(2)}\\,\\mathrm{s}\\) 附近第 ${frameIndex} 帧的听觉掩蔽范围与保护扰动。`
        : '当前显示指定时间附近的单帧听觉掩蔽曲线。'
      : '该图将整段音频各时刻的频谱取平均，展示不同频率下听觉掩蔽范围与保护扰动的关系。'

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
      setSliceError('暂未获取音频时长，无法指定时间帧。')
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
      const message = '指定时间的听觉分析曲线加载失败，请稍后重试。'
      setSliceError(message)
      pushToast({ kind: 'error', title: '加载失败', description: error instanceof Error ? error.message : message })
    } finally {
      setSliceLoading(false)
    }
  }

  return (
    <>
      <section className="flex h-full min-h-0 flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
        <div className="flex items-center justify-between gap-4 max-md:flex-wrap">
          <div className="flex items-center gap-2">
            <SectionTitle>听觉掩蔽与扰动分析</SectionTitle>
            <MetricInfoButton title="听觉掩蔽与扰动分析">{modeDescription}</MetricInfoButton>
          </div>
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
          <LineChart key={`${result.taskId}:${psychoMode}:${frameIndex ?? 'mean'}:${chartPoints.length}`} result={result} large pointsOverride={chartPoints} />
        </div>
        {sliceError && !timeDialogOpen ? <p className="mt-2 text-[11px] text-rose-300">{sliceError}</p> : null}
      </section>

      {timeDialogOpen ? createPortal(
        <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="选择听觉分析时间点">
          <form
            className="ui-card w-full max-w-[440px] !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]"
            onSubmit={(event) => {
              event.preventDefault()
              void confirmFrameTime()
            }}
          >
            <div className="mb-5 flex items-center justify-between gap-4">
              <h3 className="text-[20px] font-black text-white">选择听觉分析时间点</h3>
              <button type="button" onClick={() => setTimeDialogOpen(false)} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="取消">
                <X className="h-4 w-4" />
              </button>
            </div>
            <label className="text-[12px] font-bold text-slate-300" htmlFor="psycho-time-sec">
              时间 <MathText formula="t" className="mx-0.5 align-[-1px]" />（秒）
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
              请输入 0 到 {optionalNumber(audioDurationSec)?.toFixed(2) ?? '未生成'} 秒之间的时间，系统将显示最接近该时刻的频谱。
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
        </div>,
        document.body,
      ) : null}
    </>
  )
}

function PsychoacousticModeDropdown({ label, open, onToggle, onMean, onFrame }: { label: ReactNode; open: boolean; onToggle: () => void; onMean: () => void; onFrame: () => void }) {
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
        <div className="absolute left-1/2 top-11 z-20 w-[180px] -translate-x-1/2 rounded-[8px] border border-cyan-300/18 bg-slate-950 p-1 shadow-[0_18px_45px_rgba(0,0,0,0.42)]" role="menu">
          <button type="button" onClick={onMean} className="block h-9 w-full rounded-[6px] px-3 text-left text-[12px] font-bold text-slate-200 hover:bg-cyan-300/[0.08] hover:text-cyan-100" role="menuitem">
            <MathText formula="t" className="mr-1 align-[-1px]" />平均聚合
          </button>
          <button type="button" onClick={onFrame} className="block h-9 w-full rounded-[6px] px-3 text-left text-[12px] font-bold text-slate-200 hover:bg-cyan-300/[0.08] hover:text-cyan-100" role="menuitem">
            指定 <MathText formula="t" className="mx-1 align-[-1px]" />对应帧
          </button>
        </div>
      ) : null}
    </div>
  )
}

function QualityMetric({ label, value, tag, tone, title, foot }: { label: ReactNode; value: ReactNode; tag: string; tone: 'green' | 'blue' | 'orange'; title?: string; foot?: ReactNode }) {
  return (
    <MetricInfoSurface title={label} info={foot} tooltip={title} className="relative min-w-0 rounded-[9px] border border-cyan-300/12 bg-slate-950/16 px-3 py-3.5 text-center">
      <div className="relative min-h-7">
        <p className="absolute inset-x-0 top-1/2 min-w-0 -translate-y-1/2 truncate whitespace-nowrap px-2 text-center text-[12px] font-black text-slate-300">{label}</p>
      </div>
      <div className={cn('mt-1 flex h-6 items-center justify-center text-[20px] font-black leading-none', tone === 'green' && 'text-emerald-300', tone === 'blue' && 'text-cyan-300', tone === 'orange' && 'text-orange-300')}>{value}</div>
      <span className={cn('mt-1.5 inline-block rounded px-3 py-0.5 text-[11px] font-bold', tone === 'green' && 'bg-emerald-400/14 text-emerald-300', tone === 'blue' && 'bg-cyan-400/14 text-cyan-300', tone === 'orange' && 'bg-orange-400/14 text-orange-300')}>{tag}</span>
    </MetricInfoSurface>
  )
}

type LossDisplayKey = 'Lid' | 'Lsem' | 'Lpsy' | 'L2' | 'total'
type LossDefinition = { key: LossDisplayKey; legacyKey?: 'Lfeat'; formula: string; label: string; description: string; color: string }

const lossDescriptions: Record<LossDisplayKey, string> = {
  Lid: '声音身份目标差距',
  Lsem: '表达内容目标差距',
  Lpsy: '听感保真目标差距',
  L2: '扰动幅度目标差距',
  total: '综合目标差距',
}

const lossDefinitions: LossDefinition[] = lossTrendSeries.map((loss) => ({
  key: loss.key,
  ...('legacyKey' in loss ? { legacyKey: loss.legacyKey } : {}),
  formula: loss.formula,
  label: loss.name,
  description: lossDescriptions[loss.key],
  color: loss.color,
}))

function TrendPanel({ result, embedded }: { result: TaskResult; embedded?: boolean }) {
  const trend = downsampleTrace(result.optimizationTrace ?? result.generation?.optimizationTrace ?? result.charts.optimizationTrend)
  const lossFinal = result.lossFinal ?? result.generation?.lossFinal ?? finalLossFromTrend(trend)
  const missingLosses = lossDefinitions.filter((loss) => trend.length > 0 && trend.every((point) => lossPointValue(point, loss) === null))
  const totalIterationSteps = lastStep(trend) ?? optionalNumber(result.generation?.steps) ?? optionalNumber(result.generation?.maxSteps)
  const avgIterationSec = optionalNumber(result.averageStepSec) ?? averageStepSecFromTrace(trend) ?? (typeof result.elapsedSec === 'number' && totalIterationSteps && totalIterationSteps > 0 ? result.elapsedSec / totalIterationSteps : null)

  return (
    <section className={cn('flex min-h-0 flex-col overflow-hidden', embedded ? 'rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-5' : 'ui-card p-7')}>
      <SectionTitle>保护目标优化过程</SectionTitle>
      <div className="mt-7 grid min-h-0 grid-cols-[minmax(0,2.8fr)_minmax(240px,1fr)] gap-6 max-lg:grid-cols-1">
        <div className="relative min-h-0 max-lg:static">
          <div className="absolute inset-0 flex min-h-0 flex-col overflow-hidden rounded-[7px] border border-cyan-300/12 bg-slate-950/18 p-5 max-lg:static max-lg:min-h-[560px]">
            <div className="mb-4 flex flex-wrap items-center gap-x-8 gap-y-2 text-[11px] text-slate-400">
              {lossDefinitions.map((loss) => (
                <span key={loss.key} className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: loss.color }} />
                  <MathText formula={loss.formula} style={{ color: loss.color }} />
                </span>
              ))}
            </div>
            {trend.length > 0 ? (
              <div className="min-h-0 flex-1 overflow-hidden">
                <TrendChart data={trend} />
              </div>
            ) : (
              <div className="grid min-h-[210px] flex-1 place-items-center rounded-[6px] border border-dashed border-cyan-300/14 bg-slate-950/16 px-5 text-center text-[12px] leading-5 text-slate-400">
                暂未记录逐步优化过程，当前只能查看最终结果。
              </div>
            )}
          </div>
        </div>
        <div className="grid content-start gap-5 pr-1">
          {lossDefinitions.map((loss) => (
            <div key={loss.key} className="rounded-[7px] border border-cyan-300/12 bg-slate-950/18 px-5 py-5">
              <div className="flex items-center justify-between gap-3">
                <p className="flex items-center gap-2 text-[12px] font-bold text-slate-200">
                  <MathText formula={loss.formula} style={{ color: loss.color }} />
                  <span className="font-black" style={{ color: loss.color }}>{loss.description}</span>
                </p>
                <p className="text-[13px] font-black text-white">{formatLossNumber(lossFinalValue(lossFinal, loss))}</p>
              </div>
            </div>
          ))}
          <div className="rounded-[7px] border border-cyan-300/12 bg-slate-950/20 px-5 py-5">
            <div className="flex items-center justify-between gap-4">
              <p className="text-[11px] font-bold text-slate-400">平均每次迭代耗时</p>
              <p className="text-[14px] font-black text-cyan-200">{avgIterationSec === null ? '未生成' : <MathText formula={`${avgIterationSec.toFixed(2)}\\,\\mathrm{s/step}`} />}</p>
            </div>
            <div className="mt-3 flex items-center justify-between gap-4 border-t border-cyan-300/10 pt-3">
              <p className="text-[11px] font-bold text-slate-400">总共迭代步数</p>
              <p className="text-[14px] font-black text-white">{totalIterationSteps === null ? '未生成' : <MathText formula={`${Math.round(totalIterationSteps)}\\,\\mathrm{steps}`} />}</p>
            </div>
          </div>
        </div>
      </div>
      {missingLosses.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
          {missingLosses.map((loss) => <span key={loss.key}>{loss.label}：暂未生成</span>)}
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
        {title}
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

function DownloadModal({ result, onClose }: { result: TaskResult; onClose: () => void }) {
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)

  const runDownload = async () => {
    try {
      const file = await downloadProtectedAudio(result.taskId)
      downloadBlob(file.blob, file.filename)
      pushToast({ kind: 'success', title: '下载已开始', description: file.filename })
    } catch (error) {
      pushToast({ kind: 'error', title: '导出暂不可用', description: error instanceof Error ? error.message : '请稍后重试。' })
    }
  }

  return createPortal(
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="下载与导出">
      <div className="ui-card w-full max-w-[520px] !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-5 flex items-center justify-between gap-4">
          <div>
            <h3 className="text-[20px] font-black text-white">{result.verdict || '防护结果已生成'}</h3>
            <p className="mt-1 text-xs text-slate-500">点击此处下载</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭下载与导出">
            <X className="h-4 w-4" />
          </button>
        </div>
        <button onClick={() => void runDownload()} className="cyan-button flex h-12 w-full items-center justify-center gap-2 rounded-[8px] text-[16px] font-black">
          <Download className="h-4 w-4" />
          下载保护音频
        </button>
        <button
          onClick={() => navigate('/workspace')}
          className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-[8px] border border-cyan-300/12 bg-white/[0.035] text-[16px] font-bold text-slate-300"
        >
          <RefreshCw className="h-4 w-4" />
          重新执行任务
        </button>
      </div>
    </div>,
    document.body,
  )
}

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '未生成'
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)}MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)}KB`
  return `${bytes}B`
}

function formatMetricValue(value: unknown, type: 'percent' | 'db' | 'seconds' | 'loss' | 'bytes' | 'number') {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  if (type === 'percent') return `${(numberValue <= 1 ? numberValue * 100 : numberValue).toFixed(2)}%`
  if (type === 'db') return `${numberValue.toFixed(2)} dB`
  if (type === 'seconds') return `${numberValue.toFixed(2)} s`
  if (type === 'loss') return formatLossNumber(numberValue)
  if (type === 'bytes') return formatFileSize(numberValue)
  return numberValue.toFixed(2)
}

function friendlyAsrFailure(reason?: string | null) {
  const value = String(reason || '').trim()
  if (!value) return '模型未返回可用转写文本，请重新测试。'
  if (/cannot import name ['"]pipeline['"].*transformers/i.test(value)) return '该语音识别模型暂时无法启动，本次没有生成转写文本，请重新测试或更换模型。'
  if (/out of memory|cuda.*memory/i.test(value)) return '显存不足，模型未能完成转写，请稍后重新测试。'
  if (/missing|not found|no such file/i.test(value)) return '模型文件不完整或不可用，请检查模型部署后重新测试。'
  return value.split('\n')[0]
}

function formatRatioPercent(value: unknown) {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  return `${(numberValue * 100).toFixed(2)}%`
}

function formatTaskTime(value?: string | null) {
  if (!value) return '时间未记录'
  const dotted = value.trim().match(/^(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?$/)
  const date = dotted
    ? new Date(Number(dotted[1]), Number(dotted[2]) - 1, Number(dotted[3]), Number(dotted[4] ?? 0), Number(dotted[5] ?? 0), Number(dotted[6] ?? 0))
    : new Date(value)
  if (Number.isNaN(date.getTime())) return value
  const pad = (input: number) => String(input).padStart(2, '0')
  return `${date.getFullYear()}.${date.getMonth() + 1}.${date.getDate()} ${date.getHours()}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

function optionalNumber(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function metricStatusAvailable(status?: string | null) {
  return ['available', 'computed', 'complete', 'completed', 'success', 'partial'].includes(String(status ?? '').toLowerCase())
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function metricSource(result: TaskResult | null | undefined, keys: string[]) {
  if (!result) return undefined
  for (const key of keys) {
    const source = result.metricSources?.[key]
    if (source) return source
  }
  return undefined
}

function metricReason(result: TaskResult | null | undefined, keys: string[]) {
  const reason = metricSource(result, keys)?.reason
  return reason ? shortMetricReason(reason) : ''
}

function shortMetricReason(reason: string) {
  if (/torchcodec|libtorchcodec|FFmpeg/i.test(reason)) return '音频分析组件暂时不可用，请检查运行环境后重试'
  if (/local cache|Hub|connection|Internet|from_pretrained|huggingface/i.test(reason)) {
    if (/semantic|encoder|hubert|whisper|tokenizer|s3/i.test(reason)) return '表达内容分析模型暂时无法加载'
    if (/speaker|ecapa|speechbrain|spkrec/i.test(reason)) return '声音身份分析模型暂时无法加载'
    return '所需模型暂时无法加载'
  }
  const pesqSampleRate = /PESQ supports 8000 or 16000 Hz, got (\d+)/i.exec(reason)
  if (pesqSampleRate) return `PESQ 仅支持 8k/16k，当前 ${pesqSampleRate[1]} Hz`
  if (/confidence calibrator/i.test(reason) || /calibrated clone confidence/i.test(reason)) return '未配置克隆置信度校准模型'
  return reason.split('\n')[0].trim()
}

function friendlyCloneMetricReason(reason?: string | null) {
  const value = String(reason ?? '').trim()
  if (!value) return '待完成对应测试'
  if (/baseline|基线/i.test(value)) {
    if (/identity|speaker|声纹|身份/i.test(value)) return '原始克隆语音的声音身份参考不足'
    if (/semantic|text|transcript|语义|文本/i.test(value)) return '原始克隆语音的文本结果不足'
    if (/quality|mos|质量/i.test(value)) return '原始克隆语音的质量结果不足'
    return '原始克隆语音结果不足'
  }
  if (/dnsmos|DNSMOS|no[-_ ]?reference|无参考|quality model|quality_model/i.test(value)) return '语音质量评分暂不可用'
  if (/clone.*asr|asr.*clone|transcri|自动转写/i.test(value)) return '克隆语音文本暂未生成'
  return shortMetricReason(value)
}

function formatLossNumber(value: unknown) {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  return numberValue.toFixed(2)
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

const lossTrendFormulas: Record<LossDisplayKey, string> = {
  Lid: '\\(L_{\\mathrm{id}}\\)',
  Lsem: '\\(L_{\\mathrm{sem}}\\)',
  Lpsy: '\\(L_{\\mathrm{psy}}\\)',
  L2: '\\(L_2\\)',
  total: '\\(L_{\\mathrm{total}}\\)',
}

const lossTrendLabels: Record<LossDisplayKey, string> = {
  Lid: '声音身份目标差距',
  Lsem: '表达内容目标差距',
  Lpsy: '听感保真目标差距',
  L2: '扰动幅度目标差距',
  total: '综合目标差距',
}

function joinLossTrendFormulas(keys: LossDisplayKey[]) {
  const formulas = keys.map((key) => lossTrendFormulas[key])
  if (formulas.length <= 1) return formulas[0] ?? ''
  if (formulas.length === 2) return `${formulas[0]} 与 ${formulas[1]}`
  return `${formulas.slice(0, -1).join('、')} 与 ${formulas.at(-1)}`
}

function lossTrendConclusion(key: LossDisplayKey, direction: TrendDirection) {
  if (direction === 'insufficient') return '优化记录较少，暂不判断整体趋势。'
  if (key === 'Lid' || key === 'Lsem') {
    if (direction === 'up') return '整体上升，保护目标仍在持续变化。'
    if (direction === 'down') return '整体下降，正在稳\u2060定\u2060收\u2060敛。'
    return '整体稳定，优化过程较为平稳。'
  }
  if (key === 'Lpsy') {
    if (direction === 'up') return '整体上升，属于正\u2060常\u2060现\u2060象。'
    if (direction === 'down') return '整体下降，心理声学匹配正在改善。'
    return '整体稳定，心理声学保真过程较为平稳。'
  }
  if (key === 'L2') {
    if (direction === 'up') return '整体上升，属于正\u2060常\u2060现\u2060象。'
    if (direction === 'down') return '整体下降，扰动能量正在得到有效约束。'
    return '整体稳定，扰动能量控制较为平稳。'
  }
  if (direction === 'up') return '整体上升，优化过程仍在持续变化。'
  if (direction === 'down') return '整体下降，正在稳\u2060定\u2060收\u2060敛。'
  return '整体稳定，已进入稳\u2060定\u2060收\u2060敛阶段。'
}

function groupedLossTrendItems(trends: Record<LossDisplayKey, { direction: TrendDirection }>) {
  const sections: LossDisplayKey[][] = [['Lid', 'Lsem', 'total'], ['Lpsy', 'L2']]
  const items: string[] = []
  sections.forEach((section) => {
    const conclusionGroups = new Map<string, LossDisplayKey[]>()
    section.forEach((key) => {
      const conclusion = lossTrendConclusion(key, trends[key].direction)
      conclusionGroups.set(conclusion, [...(conclusionGroups.get(conclusion) ?? []), key])
    })
    conclusionGroups.forEach((keys, conclusion) => {
      const label = keys.length > 1
        ? keys.map((key) => lossTrendLabels[key]).join('、')
        : lossTrendLabels[keys[0]]
      items.push(`${label}：由图所示，${joinLossTrendFormulas(keys)} ${conclusion}`)
    })
  })
  return items
}

type ProtectionEvaluationContext = {
  linkedTaskStatus?: TaskStatusResponse
  asrEval?: AsrEval | null
  cloneEval?: CloneEval | null
  asrHistory?: AsrEvalResponse[]
  cloneHistory?: CloneVoiceResult[]
}

function averageAvailable(values: Array<number | null>) {
  const available = values.filter((value): value is number => value !== null && Number.isFinite(value))
  return available.length ? available.reduce((sum, value) => sum + value, 0) / available.length : null
}

function protectionQualityWeightAdvice(score: number | null, qualityNotes: string[]) {
  const prefix = `听感质量相关调参建议：${qualityNotes.length ? qualityNotes.join('') : '听感质量指标尚未完整生成。'}`
  if (score === null) {
    return `${prefix}保护听感质量评分尚未生成，暂不调整 \\(\\lambda_{\\mathrm{psy}}\\) 与 \\(\\lambda_2\\)。`
  }
  const scoreText = score.toFixed(2)
  if (score >= 85) {
    return `${prefix}保护听感质量评分为 ${scoreText} 分，说明综合听感质量优秀；当前可保持 \\(\\lambda_{\\mathrm{psy}}\\) 与 \\(\\lambda_2\\)。`
  }
  if (score >= 70) {
    return `${prefix}保护听感质量评分为 ${scoreText} 分，说明综合听感质量中等；建议适当提高 \\(\\lambda_{\\mathrm{psy}}\\) 与 \\(\\lambda_2\\)，加强心理声学保真并约束扰动能量。`
  }
  return `${prefix}保护听感质量评分为 ${scoreText} 分，说明综合听感质量较差；建议优先明显提高 \\(\\lambda_{\\mathrm{psy}}\\) 与 \\(\\lambda_2\\)，降低扰动对听感的影响后重新保护。`
}

function iterationStepsOptimizationAdvice(convergence: ReturnType<typeof analyzeLossConvergence>) {
  if (convergence.status === 'unconverged') {
    const activeLossText = joinLossTrendFormulas(convergence.active_losses)
    return `迭代步数优化建议：如图所示，训练后期${activeLossText ? ` ${activeLossText}` : ''} 曲线仍较陡峭，应当增大迭代次数 steps。`
  }
  if (convergence.status === 'converged') {
    return '迭代步数优化建议：如图所示，训练后期曲线平滑，当前迭代步数应当保持。'
  }
  return '迭代步数优化建议：当前优化记录不足，暂不判断高迭代次数阶段是否稳定收敛。'
}

function linkedAsrTuningAdvice({ linkedTaskStatus, asrEval, asrHistory }: ProtectionEvaluationContext) {
  const tasks = [...(linkedTaskStatus?.asrTasks ?? []), ...(linkedTaskStatus?.asrTask ? [linkedTaskStatus.asrTask] : [])]
  const linkedResult = tasks.find((item) => item.asrResult?.asr)?.asrResult ?? linkedTaskStatus?.asrResult
  const historyEvaluations = (asrHistory ?? [])
    .map((item) => item.asr)
    .filter((item): item is AsrEval => Boolean(item) && !['unavailable', 'failed', 'error'].includes(String(item?.status ?? '')))
  const fallbackEvaluation = linkedResult?.asr ?? asrEval
  const evaluations = historyEvaluations.length
    ? historyEvaluations
    : fallbackEvaluation && !['unavailable', 'failed', 'error'].includes(String(fallbackEvaluation.status ?? ''))
      ? [fallbackEvaluation]
      : []
  const hasTask = Boolean(tasks.length || linkedResult || asrEval)

  if (!hasTask && !evaluations.length) {
    return '调参建议（ASR 联动）：暂未生成对应 ASR 任务，请先在防护工作台进行 ASR 测试，完成识别评估。'
  }
  if (!evaluations.length && tasks.some((item) => item.status === 'queued' || item.status === 'running')) {
    return '调参建议（ASR 联动）：对应 ASR 任务正在执行，完成后将根据 WER/CER 自动判断是否需要提高 \\(\\lambda_{\\mathrm{sem}}\\)。'
  }
  if (!evaluations.length && tasks.some((item) => item.status === 'failed' || item.status === 'error')) {
    return '调参建议（ASR 联动）：对应 ASR 任务执行失败，请先重新运行 ASR 测试，再根据真实 WER/CER 调整语义权重。'
  }
  if (!evaluations.length && tasks.some((item) => item.status === 'cancelled')) {
    return '调参建议（ASR 联动）：对应 ASR 任务已取消，请先重新运行 ASR 测试。'
  }
  if (!evaluations.length) {
    return '调参建议（ASR 联动）：对应 ASR 任务暂未返回可用评估结果，请检查 ASR 模型后重试。'
  }

  const metrics = evaluations.map((evaluation) => {
    const level = evaluation.metricLevel === 'word' || evaluation.metricLevel === 'char'
      ? evaluation.metricLevel
      : chooseEditLevel(evaluation.referenceText ?? evaluation.originalText ?? '', evaluation.protectedText ?? '')
    const wer = optionalNumber(evaluation.wer)
    const cer = optionalNumber(evaluation.cer)
    return {
      name: level === 'char' && cer !== null ? 'CER' : wer !== null ? 'WER' : 'CER',
      value: level === 'char' ? cer ?? wer : wer ?? cer,
    }
  }).filter((item): item is { name: string; value: number } => item.value !== null)
  const metric = averageAvailable(metrics.map((item) => item.value))
  const metricNames = new Set(metrics.map((item) => item.name))
  const metricName = metricNames.size === 1 ? metrics[0]?.name ?? 'WER/CER' : 'WER/CER 识别错误率'
  if (metric === null) {
    return '语义相关参数调参建议（ASR 平均联动）：现有 ASR 任务均未返回可用 WER/CER，暂不据此调整 \\(\\lambda_{\\mathrm{sem}}\\)。'
  }

  const metricText = formatRatioPercent(metric)
  const sampleText = `基于 ${metrics.length} 次可用 ASR 结果，平均 ${metricName} 为 ${metricText}`
  if (metric < asrWeakDisruptionThreshold) {
    return `语义相关参数调参建议（ASR 平均联动）：${sampleText}，整体语义干扰较弱；建议提高 \\(\\lambda_{\\mathrm{sem}}\\) 后重新保护并复测。`
  }
  if (metric < asrStrongDisruptionThreshold) {
    return `语义相关参数调参建议（ASR 平均联动）：${sampleText}，整体语义干扰已有一定效果；若优先阻断语义链路，可小幅提高 \\(\\lambda_{\\mathrm{sem}}\\)。`
  }
  return `语义相关参数调参建议（ASR 平均联动）：${sampleText}，整体语义干扰效果较明显，当前可保持 \\(\\lambda_{\\mathrm{sem}}\\)。`
}

function linkedCloneTuningAdvice({ linkedTaskStatus, cloneEval, cloneHistory }: ProtectionEvaluationContext) {
  const tasks = [...(linkedTaskStatus?.cloneTasks ?? []), ...(linkedTaskStatus?.cloneTask ? [linkedTaskStatus.cloneTask] : [])]
  const linkedResult = tasks.find((item) => item.cloneResult?.cloneEval)?.cloneResult ?? linkedTaskStatus?.cloneResult
  const historyEvaluations = (cloneHistory ?? [])
    .map((item) => item.cloneEval ?? cloneResultToEval(item))
    .filter((item): item is CloneEval => Boolean(item) && !['unavailable', 'failed', 'error'].includes(String(item?.status ?? '')))
  const fallbackEvaluation = linkedResult?.cloneEval ?? cloneResultToEval(linkedResult ?? undefined) ?? cloneEval
  const evaluations = historyEvaluations.length
    ? historyEvaluations
    : fallbackEvaluation && !['unavailable', 'failed', 'error'].includes(String(fallbackEvaluation.status ?? ''))
      ? [fallbackEvaluation]
      : []
  const hasTask = Boolean(tasks.length || linkedResult || cloneEval)

  if (!hasTask && !evaluations.length) {
    return '调参建议（克隆联动）：暂未生成对应克隆任务，请先在防护工作台进行语音克隆测试，完成声音身份评估。'
  }
  if (!evaluations.length && tasks.some((item) => item.status === 'queued' || item.status === 'running')) {
    return '调参建议（克隆联动）：对应克隆任务正在执行，完成后将根据保护后声纹相似度自动判断是否需要提高 \\(\\lambda_{\\mathrm{id}}\\)。'
  }
  if (!evaluations.length && tasks.some((item) => item.status === 'failed' || item.status === 'error')) {
    return '调参建议（克隆联动）：对应克隆任务执行失败，请先重新运行语音克隆测试，再根据真实声纹相似度调整身份权重。'
  }
  if (!evaluations.length && tasks.some((item) => item.status === 'cancelled')) {
    return '调参建议（克隆联动）：对应克隆任务已取消，请先重新运行语音克隆测试。'
  }
  if (!evaluations.length) {
    return '调参建议（克隆联动）：对应克隆任务暂未返回可用声音身份评估，请检查克隆或说话人模型后重试。'
  }

  const validEvaluations = evaluations.filter((evaluation) => optionalNumber(evaluation.protectedSimilarity) !== null)
  const originalSimilarity = averageAvailable(validEvaluations.map((evaluation) => optionalNumber(evaluation.originalSimilarity)))
  const protectedSimilarity = averageAvailable(validEvaluations.map((evaluation) => optionalNumber(evaluation.protectedSimilarity)))
  const similarityDropRate = averageAvailable(validEvaluations.map((evaluation) => optionalNumber(evaluation.similarityDropRate)))
  if (protectedSimilarity === null) {
    return '声音身份相关参数调参建议（克隆平均联动）：现有克隆任务均未返回保护后声纹相似度，暂不据此调整 \\(\\lambda_{\\mathrm{id}}\\)。'
  }

  const similarityText = protectedSimilarity.toFixed(2)
  const sampleText = `基于 ${validEvaluations.length} 次可用克隆结果，保护后声纹相似度平均值为 ${similarityText}`
  const dropText = similarityDropRate === null ? '' : `，相对原始克隆的平均下降幅度为 ${formatRatioPercent(similarityDropRate)}`
  if (originalSimilarity !== null && originalSimilarity < speakerSameIdentityThreshold) {
    return `声音身份相关参数调参建议（克隆平均联动）：基于 ${validEvaluations.length} 次可用克隆结果，原始克隆声纹相似度平均值为 ${originalSimilarity.toFixed(2)}，原始克隆效果偏弱；建议先更换克隆模型或样本复测，暂不据此调整 \\(\\lambda_{\\mathrm{id}}\\)。`
  }
  if (protectedSimilarity >= speakerHighSimilarityThreshold) {
    return `声音身份相关参数调参建议（克隆平均联动）：${sampleText}${dropText}，整体声音身份残留较高；建议提高 \\(\\lambda_{\\mathrm{id}}\\) 后重新保护并复测。`
  }
  if (protectedSimilarity >= speakerSameIdentityThreshold) {
    return `声音身份相关参数调参建议（克隆平均联动）：${sampleText}${dropText}，整体仍有一定声音身份特征残留；建议小幅提高 \\(\\lambda_{\\mathrm{id}}\\)。`
  }
  return `声音身份相关参数调参建议（克隆平均联动）：${sampleText}${dropText}，整体声音身份相似度已明显降低，当前可保持 \\(\\lambda_{\\mathrm{id}}\\)。`
}

function generateProtectionInsights(result: TaskResult, evaluationContext: ProtectionEvaluationContext = {}) {
  const perturbation = result.perturbation
  const quality = result.protectionQuality ?? result.quality
  const trace = downsampleTrace(result.optimizationTrace ?? result.generation?.optimizationTrace ?? result.charts?.optimizationTrend ?? [])
  const snr = optionalNumber(perturbation?.snr) ?? optionalNumber(quality?.snr)
  const pesq = optionalNumber(quality?.pesq)
  const stoi = optionalNumber('stoi' in (quality ?? {}) ? (quality as { stoi?: number | null }).stoi : null)
  const dnsMos = optionalNumber('dnsMos' in (quality ?? {}) ? (quality as { dnsMos?: number | null }).dnsMos : null)
  const qualityScore = optionalNumber('qualityScore' in (quality ?? {}) ? (quality as { qualityScore?: number | null }).qualityScore : null)
    ?? optionalNumber(result.protectionEvaluation?.dimensions.find((dimension) => dimension.key === 'protectionQuality')?.score)
  const epsilonUsageRate = resolveEpsilonUsageRate(perturbation)
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
    const usage = formatRatioPercent(epsilonUsageRate)
    const strength = epsilonUsageRate < 0.7 ? '保护强度较为保守' : epsilonUsageRate < 0.9 ? '保护强度处于中等水平' : '当前保护强度较高'
    items.push(`指标概览：本次保护的扰动预算使用率为 ${usage}，${strength}。`)
  }

  const qualityNotes: string[] = []
  if (snr !== null) {
    const level = snr >= 25 ? '整体噪声较少' : snr >= 18 ? '整体噪声中等' : '整体噪声较明显'
    qualityNotes.push(`SNR 为 ${snr.toFixed(2)} dB，${level}。`)
  }
  if (pesq !== null) {
    const level = pesq >= 3 ? '语音感知质量良好' : pesq >= 2 ? '语音感知质量中等' : '语音感知质量较弱'
    qualityNotes.push(`PESQ 为 ${pesq.toFixed(2)}，${level}。`)
  }
  if (stoi !== null) {
    const level = stoi >= 0.9 ? '语音可懂度良好' : stoi >= 0.75 ? '语音可懂度中等' : '语音可懂度较弱'
    qualityNotes.push(`STOI 为 ${stoi.toFixed(2)}，${level}。`)
  }
  if (dnsMos !== null) {
    const level = dnsMos >= 4 ? '语音质量较好' : dnsMos >= 3 ? '语音质量中等' : '语音质量偏低'
    qualityNotes.push(`语音质量评分为 ${dnsMos.toFixed(2)}，${level}。`)
  }
  items.push(...groupedLossTrendItems(trends))
  items.push(iterationStepsOptimizationAdvice(convergence))
  items.push(protectionQualityWeightAdvice(qualityScore, qualityNotes))

  const tuning: string[] = []
  if (trends.Lid.direction === 'up' && trends.Lsem.direction === 'up') {
    tuning.push('由图所示，身份保护与语义保护曲线整体呈上升趋势。若需要强化对应保护，可适当提高 \\(\\lambda_{\\mathrm{id}}\\) 与 \\(\\lambda_{\\mathrm{sem}}\\)。')
  } else {
    if (trends.Lid.direction === 'up') tuning.push('可适当提高 \\(\\lambda_{\\mathrm{id}}\\) 强化声音身份保护。')
    if (trends.Lsem.direction === 'up') tuning.push('可适当提高 \\(\\lambda_{\\mathrm{sem}}\\) 强化语义保护。')
  }
  if (tuning.length) items.push(`身份与语义相关调参建议：${tuning.join('')}`)
  items.push(linkedAsrTuningAdvice(evaluationContext))
  items.push(linkedCloneTuningAdvice(evaluationContext))
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
  if (points.length === 0) {
    return <div className="grid h-full place-items-center text-xs text-slate-500">暂未生成听觉掩蔽频谱数据</div>
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
            {formatFrequency(point.frequency)}
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
  if (hz >= 1000) return `${(hz / 1000).toFixed(2)}k`
  return hz.toFixed(2)
}
