import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  CheckCircle2,
  ChevronDown,
  ClipboardList,
  Clock3,
  Copy,
  Download,
  Info,
  Loader2,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  TestTube2,
  Volume2,
  X,
} from 'lucide-react'
import { cloneVoice, downloadProtectedAudio, getPsychoacousticSlice, getTaskResult, getTaskStatus, listTasks, runAsrEval } from '@/services/apiClient'
import { useCapabilitiesQuery } from '@/hooks/useCapabilitiesQuery'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { AsrEval, AsrEvalResponse, AsrMetrics, CapabilitiesResponse, CloneEval, CloneVoiceRequest, CloneVoiceResult, DiffOp, EvaluationBatch, LossFinal, LossTrendPoint, ProtectionRuntimeConfig, PsychoacousticPoint, PsychoacousticSliceResponse, RadarPoint, RuntimeModelOption, SubtaskStatusSnapshot, TaskResult, TaskStatusResponse } from '@/types/task'
import type { AudioFileMeta } from '@/types/audio'
import { downloadBlob } from '@/utils/download'
import { cn } from '@/lib/utils'
import { AudioPlayer } from '@/components/audio/AudioPlayer'
import { formatDurationSeconds, getAudioDuration, getAudioSource } from '@/utils/audio'
import { TrendChart } from '@/components/charts/TrendChart'
import { MathText } from '@/components/common/MathText'
import { ModelInformationModal } from '@/components/common/ModelInformationModal'
import { computeAbsoluteDrop, formatCloneMetricNumber, generateCloneMetricInsights } from '@/utils/cloneMetricDisplay'
import { analyzeLossConvergence, analyzeLossTrend, type TrendDirection } from '@/utils/resultMetrics'
import { seconds } from '@/utils/format'

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

const defaultCloneText = 'This test shows how VoiceShield protects a speaker\'s voice.'

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
  const embeddingDistance = optionalNumber(cloneEval.embeddingDistanceAfter)
  const directSimilarity = optionalNumber(cloneEval.directSimilarity)
  if (embeddingDistance !== null && directSimilarity !== null) {
    const calibratedDistance = clamp(embeddingDistance * (1.26 - 0.3 * embeddingDistance), 0, 1)
    const directShift = 1 - directSimilarity
    return 100 * (0.9 * calibratedDistance + 0.1 * clamp(directShift / 2, 0, 1))
  }
  return optionalNumber(cloneEval.cloneDefenseScore)
}

function summarizeCloneDefenseScore(result: TaskResult, status?: TaskStatusResponse) {
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
  const { data: linkedTaskStatus } = useQuery({
    queryKey: ['task-linked-evaluations', result.taskId],
    queryFn: () => getTaskStatus(result.taskId),
    retry: false,
    refetchInterval: (query) => (hasRunningLinkedEvaluation(query.state.data) ? 1500 : false),
  })
  const cloneScore = summarizeCloneDefenseScore(result, linkedTaskStatus)
  const hasCloneScore = cloneScore.score !== null
  const cloneScoreText = cloneScore.score === null ? '未生成' : `${cloneScore.score.toFixed(2)} 分`

  return (
    <section className="ui-card grid min-h-[74px] grid-cols-[250px_180px_250px_170px_230px_minmax(290px,1fr)] items-center px-5 max-2xl:grid-cols-[1.05fr_0.78fr_1.08fr_0.75fr_1fr_1.55fr] max-xl:h-auto max-xl:grid-cols-3 max-xl:gap-y-4 max-xl:py-4">
      <SummaryItem icon={<ClipboardList />} label="保护任务" value={result.taskId} copy buttonTitle="查看任务信息" onClick={onTaskInfoClick} />
      <SummaryItem icon={<ShieldCheck />} label="保护状态" value={statusText[result.status] ?? result.status} green={result.status === 'completed' || result.status === 'success'} />
      <SummaryItem icon={<Clock3 />} label="完成时间" value={result.completedAt ?? '-'} />
      <SummaryItem icon={<Clock3 />} label="处理耗时" value={typeof result.elapsedSec === 'number' ? formatElapsed(result.elapsedSec) : '-'} />
      <SummaryItem icon={<Sparkles />} label="防护模式" value={modeText[result.mode] ?? result.mode} green />
      <button
        type="button"
        onClick={onDownloadClick}
        title={hasCloneScore ? `基于 ${cloneScore.count} 次可用克隆评估的平均分` : '等待语音克隆测试'}
        className="flex h-full min-h-[58px] items-center justify-center gap-3 border-l border-cyan-300/10 pl-5 transition hover:bg-cyan-400/[0.035]"
      >
        {hasCloneScore ? <ShieldCheck className="h-11 w-11 text-cyan-300" /> : null}
        <p className={cn('shrink-0 font-mono text-[27px] font-black leading-none', hasCloneScore ? 'text-emerald-300' : 'text-rose-300')}>
          {cloneScoreText}
        </p>
        <div className="min-w-0 text-left">
          <p className="truncate text-[16px] font-black leading-tight text-cyan-100">{hasCloneScore ? '保护结果已生成' : '等待克隆测试'}</p>
          <p className="mt-1 truncate text-xs text-slate-400">{hasCloneScore ? '点击此处下载' : '完成评估后自动更新'}</p>
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

function MetricInfoButton({ title, children }: { title: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open])

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
      {open ? createPortal(
        <div className="fixed inset-0 z-[240] grid place-items-center bg-slate-950/80 px-4" role="dialog" aria-modal="true" aria-label="指标说明" onClick={() => setOpen(false)}>
          <div className="ui-card w-full max-w-[520px] !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.56)]" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-bold tracking-[0.12em] text-cyan-300">指标说明</p>
                <h3 className="mt-2 text-lg font-black text-white">{title}</h3>
              </div>
              <button type="button" onClick={() => setOpen(false)} className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-cyan-300/14 text-slate-300 hover:text-white" aria-label="关闭指标说明">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-5 rounded-[8px] border border-cyan-300/12 bg-slate-950 p-4 text-sm leading-7 text-slate-300">{children}</div>
          </div>
        </div>,
        document.body,
      ) : null}
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
  const [cloneForm, setCloneForm] = useState<CloneVoiceRequest>({
    text: result.asr.originalText || defaultCloneText,
    model: result.cloneResults?.at(-1)?.request?.model ?? '',
    language: initialEvaluationLanguage,
    speed: 1,
    speakerPrompt: result.asr.originalText || '',
    annotationSource: 'manual',
  })
  const { data: capabilities } = useCapabilitiesQuery()
  const { data: linkedTaskStatus, refetch: refetchLinkedTaskStatus } = useQuery({
    queryKey: ['task-linked-evaluations', result.taskId],
    queryFn: () => getTaskStatus(result.taskId),
    retry: false,
    refetchInterval: (query) => (hasRunningLinkedEvaluation(query.state.data) ? 1500 : false),
  })
  const runtimeConfig = configFromCapabilities(capabilities)
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
  const ttsModelOptions = useMemo(
    () => {
      if (configuredTtsOptions.length) return configuredTtsOptions
      const value = result.cloneResults?.at(-1)?.request?.model
      return value ? [{ label: value, name: value, value, status: 'available' as const }] : []
    },
    [configuredTtsOptions, result.cloneResults],
  )
  const ttsOptions = useMemo(() => ttsModelOptions.filter(isAvailableModel).map((option) => option.value), [ttsModelOptions])
  const selectedTtsOption = ttsModelOptions.find((option) => option.value === cloneForm.model) ?? ttsModelOptions[0]
  const evaluationModel = useMemo(() => backendOptionItems(runtimeConfig?.models.evaluation)[0] ?? null, [runtimeConfig?.models.evaluation])
  const oneClickAsrLimit = Math.max(1, asrOptions.filter(isAvailableModel).length)
  const oneClickCloneLimit = Math.max(1, ttsModelOptions.filter(isAvailableModel).reduce((count, option) => count + (option.promptRequired ? 2 : 1), 0))
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
      const snapshotResult = snapshot.cloneResult ?? current?.result
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
  const latestCompletedCloneEntry = [...cloneHistory].reverse().find((item) => item.result)
  const activeCloneEntry = selectedCloneKey ? selectedCloneEntry : latestCompletedCloneEntry
  const activeCloneResult = activeCloneEntry?.result
  const activeCloneEval = selectedCloneKey
    ? activeCloneResult?.cloneEval ?? cloneResultToEval(activeCloneResult) ?? null
    : activeCloneResult?.cloneEval ?? cloneResultToEval(activeCloneResult) ?? result.cloneEval ?? null
  const completedCloneHistory = cloneHistory.map((item) => item.result).filter((item): item is CloneVoiceResult => Boolean(item))
  const compareTabs = [
    {
      key: 'protect',
      label: '语音保护结果',
      modelTitle: result.processingModel ?? result.generation?.source ?? '未生成',
    },
    {
      key: 'clone',
      label: '克隆测试结果',
      modelTitle: '',
    },
  ] as const
  const activeModelTitle = compareTabs.find((tab) => tab.key === activePanel)?.modelTitle ?? '未生成'

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
        return {
          ...current,
          model,
          language: modelLanguages.includes(current.language ?? '') ? current.language : nextLanguage,
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
    if (normalizeEvaluationLanguage(nextForm.language) !== normalizeEvaluationLanguage(cloneForm.language)) {
      const language = normalizeEvaluationLanguage(nextForm.language)
      setAsrLanguage(language)
      setAsrModel(preferredAsrModel(asrOptions, language))
    }
    setCloneForm(nextForm)
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
          throw new Error(asrResult.asr.error || 'ASR 测试失败，请检查模型或依赖。')
        }
        return asrResult.asr
      }
      if (asrTaskStatus === 'failed' || asrTaskStatus === 'error') {
        const taskError = asrTask?.error ?? status.error
        throw new Error(typeof taskError === 'string' ? taskError : asrTask?.message || status.message || 'ASR 测试失败，请检查服务状态。')
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

  const validateCloneRequest = (request: CloneVoiceRequest) => {
    const modelOption = ttsModelOptions.find((option) => option.value === request.model)
    const modelLanguages = modelOption?.languages?.length ? modelOption.languages : cloneLanguages
    if (!request.text.trim()) return '请输入用于语音克隆的文本。'
    if (!ttsOptions.includes(request.model) || !modelOption) return '请选择当前可用的克隆模型。'
    if (modelOption.promptRequired && request.annotationSource !== 'asr' && !request.speakerPrompt?.trim()) return '该模型需要填写一条人工标注。'
    if (modelOption.promptRequired && request.annotationSource === 'asr' && (!request.annotationAsrSubId || !request.originalSpeakerPrompt?.trim() || !request.protectedSpeakerPrompt?.trim())) return '请选择一条同时包含原始音频和保护音频转写的 ASR 标注。'
    if (!modelLanguages.includes(request.language ?? '')) return '请选择当前模型支持的克隆语言。'
    if (!cloneSpeeds.includes(Number(request.speed))) return '请选择当前支持的克隆语速。'
    return undefined
  }

  const submitCloneTest = async (requestOverride = cloneForm) => {
    const validationError = validateCloneRequest(requestOverride)
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
      const response = await cloneVoice(result.taskId, { ...requestOverride, text: requestOverride.text.trim() })
      if (response.cloneSubId) setSelectedCloneKey(`sub:${response.cloneSubId}`)
      await refetchLinkedTaskStatus()
      const nextResult =
        (response.status === 'completed' || response.status === 'success') && getAudioSource(response.originalCloneAudio) && getAudioSource(response.protectedCloneAudio)
          ? response
          : await waitForCloneResult(result.taskId, response.cloneSubId)
      setCloneResult(nextResult)
      setSelectedCloneKey(nextResult.cloneSubId ? `sub:${nextResult.cloneSubId}` : `clone:${nextResult.cloneId}`)
      await refetchLinkedTaskStatus()
      pushToast({ kind: 'success', title: '语音克隆测试完成', description: nextResult.message ?? nextResult.cloneId })
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
    const quickRequest: CloneVoiceRequest = {
      ...cloneForm,
      text: cloneForm.text.trim() || defaultCloneText,
      model: modelOption.value,
      language: requestedLanguage,
      speed: cloneSpeeds.includes(Number(cloneForm.speed)) ? cloneForm.speed : 1,
    }
    if (modelOption.promptRequired && latestAnnotation) {
      quickRequest.annotationSource = 'asr'
      quickRequest.annotationAsrSubId = latestAnnotation.asrSubId
      quickRequest.annotationAsrModel = latestAnnotation.asr?.model
      quickRequest.annotationCreatedAt = latestAnnotation.createdAt ?? undefined
      quickRequest.speakerPrompt = latestAnnotation.asr?.originalText ?? ''
      quickRequest.originalSpeakerPrompt = latestAnnotation.asr?.originalText ?? ''
      quickRequest.protectedSpeakerPrompt = latestAnnotation.asr?.protectedText ?? ''
    } else if (modelOption.promptRequired) {
      quickRequest.annotationSource = 'manual'
      quickRequest.speakerPrompt = manualPrompt
      quickRequest.originalSpeakerPrompt = undefined
      quickRequest.protectedSpeakerPrompt = undefined
      quickRequest.annotationAsrSubId = undefined
      quickRequest.annotationAsrModel = undefined
      quickRequest.annotationCreatedAt = undefined
    }
    setCloneForm(quickRequest)
    await submitCloneTest(quickRequest)
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
          <SectionTitle info>结果对比</SectionTitle>
          <div className="flex items-center gap-2">
            {compareTabs.map(({ key, label }) => (
              <button key={key} type="button" onClick={() => setActivePanel(key as ComparePanel)} className={cn('h-9 rounded-[7px] border border-cyan-300/14 px-3 text-sm font-black text-slate-300 transition hover:text-white', activePanel === key && 'bg-cyan-400/14 text-cyan-200')} title={`查看${label}结果`}>
                {label}
              </button>
            ))}
          </div>
        </div>
        {activePanel === 'protect' ? (
          <p className="pointer-events-none absolute left-1/2 max-w-[34%] -translate-x-1/2 truncate text-center text-[14px] font-black tracking-[0.01em] text-cyan-100/90" title={activeModelTitle}>
            {activeModelTitle}
          </p>
        ) : null}
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
          <CloneTab result={result} cloneResult={activeCloneResult} cloneEval={activeCloneEval} cloneHistory={cloneHistory} cloneBatches={linkedTaskStatus?.cloneBatches ?? []} selectedCloneKey={selectedCloneKey} onSelectClone={openCloneHistoryResult} onOpenAsr={openAsrHistoryResult} loading={cloneLoading} status={cloneTaskStatus} evaluationModel={evaluationModel} cloneModelOptions={ttsModelOptions} modelTypes={runtimeConfig?.modelTypes} asrEval={activeAsrEval} asrEditStats={asrEditStats} asrHistory={asrHistory} asrBatches={linkedTaskStatus?.asrBatches ?? []} selectedAsrSubId={selectedAsrSubId ?? activeAsrResult?.asrSubId} asrHistoryLimit={oneClickAsrLimit} cloneHistoryLimit={oneClickCloneLimit} />
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
        <div className="min-h-0">
          <section className="flex min-h-0 flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
            <SectionTitle>扰动与可听性分析</SectionTitle>
            <div className="mt-5 grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-3">
              <ScoreBox label={<span className="inline-flex items-center justify-center gap-0.5">扰动强度</span>} value={formatMetricValue(perturbation?.l2Norm ?? result.quality.l2Norm, 'loss')} foot="整段音频的总体改动量，数值越小表示越接近原始音频。" />
              <ScoreBox label="扰动上限利用率" value={formatMetricValue(epsilonUsageRate, 'percent')} foot="已使用的扰动预算比例，越接近 100% 表示越接近设定上限。" />
              <ScoreBox label="信噪比（SNR）" value={formatMetricValue(snr, 'db')} foot="原始语音与扰动噪声的强弱比；数值越高，音频越接近原音。" />
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
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(360px,0.72fr)] items-stretch gap-5 max-xl:grid-cols-1">
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
          text={failed && selectedHistory ? asrHistoryFailureReason(selectedHistory) : selectedHistory?.taskStatus?.message || (pending ? '任务状态会自动刷新，完成后显示完整转写与指标。' : 'ASR 评估属于可选下游测试，开始测试后显示转写差异与语义链路指标。')}
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
  const textMetricLevel = asrEval.metricLevel ?? editStats?.level ?? 'word'
  const tokenUnavailableReason = tokenDiff == null
    ? sharedSemantic?.error || sharedSemantic?.reason || metricReason(result, ['semanticEval.tokenChangeRate', 'semanticEval.tokenErrorRate', 'asrEval.tokenChangeRate', 'asrEval.tokenErrorRate'])
    : ''
  const tokenFormulaText = tokenUsesEditDistance
    ? '当前回退值为保护前后离散语音 Token 序列的编辑距离除以原始序列长度。'
    : '后端使用语义 tokenizer 编码保护前后音频，在两侧较短序列长度内统计同位置离散语音 Token 不同的比例。'
  const tokenFoot = `${tokenUnavailableReason ? `${tokenUnavailableReason}。` : ''}${tokenFormulaText}该指标不按 ASR 文本的字符或单词切分。`
  const metricLevelFoot = textMetricLevel === 'char'
    ? '中文按字/字符（char）作为 CER 与文本编辑操作的统计单位。'
    : '英文按词（word）作为 WER 与文本编辑操作的统计单位。'
  const semanticSourceInfo = metricSource(result, ['semanticEval.semanticDrift', 'asrEval.semanticDrift'])
  const semanticIsMfccProxy = String(semanticSourceInfo?.source ?? '').toLowerCase() === 'mfcc_proxy'
  const semanticFoot = sharedSemantic?.semanticDrift == null
    ? sharedSemantic?.error || sharedSemantic?.reason || metricReason(result, ['semanticEval.semanticDrift', 'asrEval.semanticDrift'])
    : semanticIsMfccProxy
      ? '1 − 平均余弦值；当前为 MFCC 声学特征代理，不等同于深度语义表示。'
      : '1 − 多个语义编码器的平均加权余弦值，越高表示语义表示变化越大。'
  const semanticDetailLabel = semanticIsMfccProxy ? 'MFCC 代理漂移' : '语义表示漂移'
  const errorShares = asrErrorShares(asrEval, editStats)
  const asrFailureReason = ['unavailable', 'failed', 'error'].includes(String(asrEval.status ?? '').toLowerCase())
    ? friendlyAsrFailure(asrEval.error || asrEval.reason)
    : null

  return (
    <div className="space-y-5">
      <p className="text-center text-sm font-black text-cyan-100">ASR 标注 · {shortAsrModelName(asrEval.model)}</p>
      <AsrHistoryPanel history={history} batches={batches} selectedAsrSubId={selectedAsrSubId} onSelect={onSelect} onOpenBatch={onOpenBatch} maxVisible={historyLimit} />
      {asrFailureReason ? <MetricNotice text={`该次 ASR 转写失败：${asrFailureReason}`} /> : null}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_288px_minmax(0,1fr)]">
        <TextBox title="参考文本 / 原始转写（ASR）" text={referenceText || '未生成'} foot="用于 WER/CER 与 diff 的参考文本" />
        <div className="grid grid-cols-2 content-center gap-3">
          <ScoreBox label="WER（词错率）" value={formatMetricValue(wer, 'percent')} red compact />
          <ScoreBox label="CER（字错率）" value={formatMetricValue(cer, 'percent')} red compact />
          <ScoreBox label="IR（插入率）" value={formatMetricValue(insertRate, 'percent')} red compact />
          <ScoreBox label="SR（替换率）" value={formatMetricValue(substituteRate, 'percent')} red compact />
        </div>
        <TextBox title="保护音频转写（ASR）" text={protectedText || '未生成'} foot="红色为新增内容，绿色删除线为原文缺失内容" content={diffOps.length ? renderDiffOps(diffOps) : undefined} />
      </div>
      <div className="grid grid-cols-[1.05fr_0.95fr] gap-5 max-lg:grid-cols-1">
        <MetricPanel title="保护任务共享语义指标">
          <ScoreBox label={semanticDetailLabel} value={formatMetricValue(sharedSemantic?.semanticDrift, 'number')} foot={semanticFoot} />
          <ScoreBox label="Token 变化率" value={formatMetricValue(tokenDiff, 'percent')} foot={tokenFoot} />
          <ScoreBox label="指标层级" value={textMetricLevel} foot={metricLevelFoot} />
        </MetricPanel>
        <RateBreakdown substituteShare={errorShares?.substituteShare} insertShare={errorShares?.insertShare} />
      </div>
      <InsightPanel title="ASR 结果解读" items={generateAsrInsights(asrEval, editStats)} naturalHeight />
    </div>
  )
}

function CloneTab({ result, cloneResult, cloneEval, cloneHistory, cloneBatches, selectedCloneKey, onSelectClone, onOpenAsr, loading, status, evaluationModel, cloneModelOptions, modelTypes, asrEval, asrEditStats, asrHistory, asrBatches, selectedAsrSubId, asrHistoryLimit, cloneHistoryLimit }: { result: TaskResult; cloneResult?: CloneVoiceResult; cloneEval?: CloneEval | null; cloneHistory: CloneHistoryEntry[]; cloneBatches: EvaluationBatch[]; selectedCloneKey?: string; onSelectClone: (cloneKey: string) => void; onOpenAsr: (asrSubId?: string) => void; loading: boolean; status: TaskStatusResponse | null; evaluationModel: BackendSelectOption | null; cloneModelOptions: BackendSelectOption[]; modelTypes?: CapabilitiesResponse['modelTypes']; asrEval?: AsrEval | null; asrEditStats: EditMetrics | null; asrHistory: AsrHistoryEntry[]; asrBatches: EvaluationBatch[]; selectedAsrSubId?: string; asrHistoryLimit: number; cloneHistoryLimit: number }) {
  const [manualAnnotation, setManualAnnotation] = useState<CloneHistoryEntry | null>(null)
  const [fineTuneReport, setFineTuneReport] = useState<CloneHistoryEntry | null>(null)
  const [informationModel, setInformationModel] = useState<BackendSelectOption | null>(null)
  const [batchDetail, setBatchDetail] = useState<EvaluationBatch | null>(null)
  const openBatch = (batch: EvaluationBatch) => {
    setManualAnnotation(null)
    setFineTuneReport(null)
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
  const asrSection = asrEval || asrHistory.length || asrBatches.length || selectedAsrSubId ? (
    <div id="asr-result-detail" className="scroll-mt-24">
      <AsrTab result={result} asrEval={asrEval} editStats={asrEditStats} history={asrHistory} batches={asrBatches} selectedAsrSubId={selectedAsrSubId} onSelect={onOpenAsr} onOpenBatch={openBatch} historyLimit={asrHistoryLimit} />
    </div>
  ) : null
  const cloneHistorySection = cloneHistory.length || cloneBatches.length ? (
    <CloneHistoryPanel
      history={cloneHistory}
      batches={cloneBatches}
      selectedCloneKey={selectedCloneKey}
      onSelect={onSelectClone}
      onOpenBatch={openBatch}
      onOpenAsr={onOpenAsr}
      onOpenManual={(item) => { setBatchDetail(null); setFineTuneReport(null); setManualAnnotation(item) }}
      onOpenFineTune={(item) => { setBatchDetail(null); setManualAnnotation(null); setFineTuneReport(item) }}
      maxVisible={cloneHistoryLimit}
    />
  ) : null
  const showLoading = loading || selectedClonePending

  let cloneDetail: ReactNode
  if (showLoading) {
    const liveStatus = selectedCloneEntry?.taskStatus ?? status
    cloneDetail = (
      <div className="grid items-center gap-6 pl-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
        <LoadingCard title="克隆原语音" progress={optionalNumber(liveStatus?.progress) ?? undefined} message={liveStatus?.message ?? undefined} />
        <div className="compare-badge mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
        <LoadingCard title="克隆保护语音" progress={optionalNumber(liveStatus?.progress) ?? undefined} message={liveStatus?.message ?? undefined} />
      </div>
    )
  } else if (selectedCloneFailed) {
    cloneDetail = <EmptyState title={selectedCloneStatus === 'cancelled' ? '语音克隆测试已取消' : '语音克隆测试失败'} text={selectedCloneError || '该次克隆任务未生成可用结果。'} />
  } else if (!cloneEval) {
    cloneDetail = <EmptyState title="未执行语音克隆测试" text="请在防护工作台选择已完成的保护任务并开始语音克隆测试。" />
  } else {
    const cloneReason = cloneEval.reason ? shortMetricReason(cloneEval.reason) : metricReason(result, ['cloneEval.*'])
    const cloneModelLabel = shortCloneModelName(cloneEval.cloneModel ?? cloneResult?.request.model)
    const speakerModelLabel = evaluationModel?.label ?? shortCloneModelName(cloneEval.speakerEvalModel ?? cloneEval.speakerModel ?? undefined)
    const cloneModelValue = cloneEval.cloneModel ?? cloneResult?.request.model
    const cloneModelOption = cloneModelOptions.find((item) => item.value === cloneModelValue || item.backendValue === cloneModelValue) ?? null
    const activeFineTuneEntry = selectedCloneEntry?.result?.fineTune ? selectedCloneEntry : cloneHistory.find((item) => item.result === cloneResult && item.result?.fineTune)
    cloneDetail = (
      <>
        <div className="flex flex-wrap items-center justify-center gap-2 text-center text-sm font-black text-violet-100">
          <span>TTS 克隆 · {cloneModelLabel}</span>
          {cloneModelOption ? <ModelInfoButton model={cloneModelOption} onOpen={setInformationModel} /> : null}
          <span>· 评估 {speakerModelLabel}</span>
          {evaluationModel ? <ModelInfoButton model={evaluationModel} onOpen={setInformationModel} /> : null}
          {activeFineTuneEntry ? <button type="button" onClick={() => setFineTuneReport(activeFineTuneEntry)} className="inline-flex h-7 items-center gap-1 rounded-[6px] border border-amber-300/18 px-2 text-[11px] font-black text-amber-200 hover:bg-amber-300/10"><Search className="h-3.5 w-3.5" />微调报告</button> : null}
        </div>
        <div id="clone-result-detail" className="scroll-mt-24 space-y-5">
          {cloneReason ? <MetricNotice text={`克隆指标未生成原因：${cloneReason}`} /> : null}
        <div className="grid items-center gap-6 pl-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
          {cloneEval.originalCloneAudio ? <AudioCard title="克隆原语音" audio={cloneEval.originalCloneAudio} color="#a78bfa" /> : <EmptyMetricCard title="克隆原语音" text="暂未生成克隆原语音" />}
          <div className="compare-badge mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
          {cloneEval.protectedCloneAudio ? <AudioCard title="克隆保护语音" audio={cloneEval.protectedCloneAudio} color="#f59e0b" /> : <EmptyMetricCard title="克隆保护语音" text="暂未生成克隆保护语音" />}
        </div>
        <div className="grid items-stretch grid-cols-[minmax(420px,0.96fr)_minmax(520px,1.04fr)] gap-5 max-xl:grid-cols-1">
          <CloneIdentityPanel cloneEval={cloneEval} evaluationModel={evaluationModel} />
          <div className="relative min-h-0 max-xl:min-h-[330px]">
            <div className="absolute inset-0 max-xl:static max-xl:h-[330px]">
              <CloneVisualizationPanel result={result} cloneEval={cloneEval} />
            </div>
          </div>
        </div>
        <div className="grid grid-cols-1 gap-5">
          <InsightPanel title="克隆结果解读" items={generateCloneInsights(cloneEval)} />
        </div>
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
      <FineTuneReportModal item={fineTuneReport} onClose={() => setFineTuneReport(null)} />
      <BatchProgressModal batch={liveBatchDetail} onClose={() => setBatchDetail(null)} />
      <ModelInformationModal model={informationModel} modelTypes={modelTypes} onClose={() => setInformationModel(null)} />
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
  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex items-start justify-between gap-3">
        <SectionTitle>错误类型占比</SectionTitle>
        <MetricInfoButton title="错误类型占比">分别统计替换与插入在全部编辑错误中的占比，用于判断保护音频主要通过哪类文本变化干扰识别。</MetricInfoButton>
      </div>
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

function CloneIdentityPanel({ cloneEval, evaluationModel }: { cloneEval: CloneEval; evaluationModel: BackendSelectOption | null }) {
  const originalSimilarity = optionalNumber(cloneEval.originalSimilarity)
  const protectedSimilarity = optionalNumber(cloneEval.protectedSimilarity)
  const embeddingDistanceBefore = optionalNumber(cloneEval.embeddingDistanceBefore)
  const embeddingDistance = optionalNumber(cloneEval.embeddingDistanceAfter)
  const similarityDrop = computeAbsoluteDrop(cloneEval.originalSimilarity, cloneEval.protectedSimilarity)
  const similarityDelta = similarityDrop === null ? null : Math.abs(similarityDrop)
  const similarityDecreased = originalSimilarity !== null && protectedSimilarity !== null ? protectedSimilarity <= originalSimilarity : true
  const embeddingDelta = embeddingDistanceBefore === null || embeddingDistance === null ? null : Math.abs(embeddingDistance - embeddingDistanceBefore)
  const embeddingIncreased = embeddingDistanceBefore !== null && embeddingDistance !== null ? embeddingDistance >= embeddingDistanceBefore : true
  const directSimilarity = optionalNumber(cloneEval.directSimilarity)
  const directShift = directSimilarity === null ? null : 1 - directSimilarity
  const defenseScore = computeCloneDefenseScore(cloneEval)

  return (
    <section className="flex flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex items-center justify-between gap-3">
        <SectionTitle>声音身份特征链路分析</SectionTitle>
        {evaluationModel ? <span className="text-xs font-bold text-cyan-200">{evaluationModel.label}</span> : null}
      </div>
      <div className="mt-4 grid min-h-0 grid-cols-2 gap-3 max-sm:grid-cols-1">
        <IdentityTransformCard
          title="说话人相似度"
          before={formatCloneMetricNumber(originalSimilarity)}
          after={formatCloneMetricNumber(protectedSimilarity)}
          delta={similarityDelta}
          changeLabel={similarityDecreased ? '下降' : '上升'}
          foot={<div className="space-y-1"><p>比较原语音与保护前后两种克隆语音的声纹相似程度。</p><p>范围为 [-1, 1]，保护后越低越好。</p></div>}
          tone={similarityDecreased ? 'green' : 'red'}
        />
        <IdentityTransformCard
          title="声纹嵌入距离"
          before={formatCloneMetricNumber(embeddingDistanceBefore)}
          after={formatCloneMetricNumber(embeddingDistance)}
          delta={embeddingDelta}
          changeLabel={embeddingIncreased ? '增加' : '减少'}
          foot={<div className="space-y-1"><p>用 1 减去克隆语音与原语音的声纹相似度。</p><p>范围为 [0, 2]，距离越大表示保护效果越好。</p></div>}
          tone={embeddingIncreased ? 'red' : 'green'}
        />
        <IdentityValueCard
          title="直接声纹偏移"
          value={formatCloneMetricNumber(directShift)}
          foot={<div className="space-y-1"><p>用 1 减去原语音与保护语音的声纹相似度。</p><p>范围为 [0, 2]，越大越好；它不直接比较克隆结果，因此只作为低权重参考。</p></div>}
          tone="green"
        />
        <IdentityValueCard
          title="保护效果评估"
          value={defenseScore === null ? '不可用' : `${defenseScore.toFixed(2)} 分`}
          foot={<div className="space-y-1"><p>以保护后克隆语音的声纹距离为主要依据，并参考少量直接声纹偏移。两项经过归一化和软映射后得到 0–100 分，分数越高表示保护效果越好。</p></div>}
          tone="green"
        />
      </div>
    </section>
  )
}

function IdentityTransformCard({ title, before, after, delta, changeLabel, foot, tone }: { title: string; before: string; after: string; delta: number | null; changeLabel: string; foot: ReactNode; tone: 'green' | 'red' }) {
  return (
    <div className="relative flex flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/22 px-4 py-5">
      <div className="relative min-h-8">
        <p className="absolute inset-x-0 top-1/2 -translate-y-1/2 px-9 text-center text-[14px] font-black leading-5 text-slate-100">{title}</p>
        <div className="absolute right-0 top-1/2 -translate-y-1/2"><MetricInfoButton title={title}>{foot}</MetricInfoButton></div>
      </div>
      <div className="my-auto grid grid-cols-[minmax(0,1fr)_30px_minmax(0,1fr)] items-center py-3 text-center font-mono text-[26px] font-black">
        <span className="text-slate-200">{before}</span>
        <span className="text-slate-500">→</span>
        <span className={tone === 'green' ? 'text-emerald-300' : 'text-rose-300'}>{after}</span>
      </div>
      <div className={cn('mb-3 rounded-[6px] border py-1.5 text-center text-sm font-black', tone === 'green' ? 'border-emerald-300/18 bg-emerald-400/10 text-emerald-300' : 'border-red-300/18 bg-red-400/10 text-red-300')}>
        {changeLabel} {delta === null ? '不可用' : delta.toFixed(2)}
      </div>
    </div>
  )
}

function IdentityValueCard({ title, value, foot, tone }: { title: string; value: string; foot: ReactNode; tone: 'green' | 'red' }) {
  return (
    <div className="relative flex flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/22 px-4 py-5">
      <div className="relative min-h-8">
        <p className="absolute inset-x-0 top-1/2 -translate-y-1/2 px-9 text-center text-[14px] font-black leading-5 text-slate-100">{title}</p>
        <div className="absolute right-0 top-1/2 -translate-y-1/2"><MetricInfoButton title={title}>{foot}</MetricInfoButton></div>
      </div>
      <p className={cn('my-auto py-5 text-center font-mono text-[34px] font-black', tone === 'green' ? 'text-emerald-300' : 'text-rose-300')}>{value}</p>
    </div>
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
        <MetricInfoButton title="说话人防护雷达图">将直接声纹偏移、相似度下降、嵌入距离增加和保护效果评估映射到统一的 0–100 分尺度；面积越大表示综合防护效果越明显。</MetricInfoButton>
      </div>
      <div className="mt-4 min-h-0 flex-1 overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-2">
        {!displayRadar.length ? (
          <ChartEmptyState text="暂未生成说话人防护雷达数据" />
        ) : availableRadar.length < 3 ? (
          <ChartEmptyState text="可用雷达指标不足，至少需要 3 个真实指标" />
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
  const defenseScore = computeCloneDefenseScore(cloneEval)
  return radar
    .filter((item) => hasRealCloneConfidence || !/置信|confidence/i.test(item.name))
    .map((item) => {
      const name = normalizeCloneRadarName(item.name)
      return {
        ...item,
        name,
        value: name === '保护效果评估' && defenseScore !== null ? defenseScore : item.value,
      }
    })
}

function normalizeCloneRadarName(name: string) {
  if (/直接|direct/i.test(name)) return '直接声纹偏移'
  if (/相似|similar/i.test(name)) return '相似度下降'
  if (/嵌入|embedding|距离/i.test(name)) return '嵌入距离增加'
  if (/保护后|防护|protected/i.test(name)) return '保护效果评估'
  if (/置信|confidence/i.test(name)) return '克隆置信度下降'
  return name
}

function CloneRadarPreview({ radar, availableRadar }: { radar: RadarPoint[]; availableRadar: RadarPoint[] }) {
  const width = 620
  const height = 340
  const centerX = width / 2
  const centerY = height / 2
  const radius = 105
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
      labelX: centerX + Math.cos(angle) * (radius + 58),
      labelY: centerY + Math.sin(angle) * (radius + 42),
    }
  })
  const polygon = axisPoints.map((point) => `${point.valueX.toFixed(1)},${point.valueY.toFixed(1)}`).join(' ')
  const missing = radar.filter((item) => !(typeof item.value === 'number' && Number.isFinite(item.value)))
  const missingNames = missing.map((item) => item.reason ? `${item.name}（${item.reason}）` : item.name).filter(Boolean)

  return (
    <div className="flex h-full flex-col">
      <div className="grid min-h-0 flex-1 place-items-center">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-full min-h-0 w-full max-w-none">
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
          <p className="text-xs font-bold text-slate-400">部分指标未生成</p>
          <ul className="mt-1 space-y-1 text-xs leading-5 text-slate-500">
            {missingNames.map((name) => <li key={name}>• {name}</li>)}
          </ul>
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
  return (
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
                  {unavailable ? <span className="mt-0.5 block truncate text-[10px] font-medium text-amber-200">{item.status}</span> : null}
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
    </div>
  )
}

function annotationSourceLabel(source?: CloneVoiceRequest['annotationSource']) {
  return source === 'asr' ? 'ASR 标注' : '人工标注'
}

function shortCloneModelName(value?: string) {
  const model = String(value ?? '')
  if (/xtts[_:/-]?v?2/i.test(model)) return 'XTTS v2'
  if (/xtts[_:/-]?v?1[._-]?1/i.test(model)) return 'XTTS v1.1'
  if (/yourtts/i.test(model)) return 'YourTTS'
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

function AsrHistoryPanel({ history, batches, selectedAsrSubId, onSelect, onOpenBatch, maxVisible }: { history: AsrHistoryEntry[]; batches: EvaluationBatch[]; selectedAsrSubId?: string; onSelect: (asrSubId?: string) => void; onOpenBatch: (batch: EvaluationBatch) => void; maxVisible: number }) {
  if (!history.length && !batches.length) return null
  const rowCount = history.length + batches.length
  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex flex-col items-center justify-center gap-1 text-center">
        <SectionTitle>同一保护任务的 ASR 任务对比</SectionTitle>
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
            <tr><th className="px-2 py-2 text-center">任务</th><th className="px-2 py-2 text-center">ASR 模型</th><th className="px-2 py-2 text-center">语言</th><th className="px-2 py-2 text-center">进度</th><th className="px-2 py-2 text-center">处理时长</th><th className="px-2 py-2 text-center">WER</th><th className="px-2 py-2 text-center">CER</th></tr>
          </thead>
          <tbody>
            {batches.map((batch) => {
              const progress = clamp(optionalNumber(batch.progress) ?? 0, 0, 1)
              const progressPercent = Math.round(progress * 100)
              const tone = progressTone(batch.status)
              const elapsed = batchElapsed(batch)
              return (
                <tr key={batch.batchId} role="button" tabIndex={0} onClick={() => onOpenBatch(batch)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onOpenBatch(batch) } }} className="cursor-pointer border-b border-violet-300/14 bg-violet-400/[0.06] hover:bg-violet-400/[0.11] focus:outline-none focus:ring-1 focus:ring-inset focus:ring-violet-300/40">
                  <td className="px-2 py-2 text-center font-mono text-violet-200" title={batch.batchId}><span className="block truncate">{batch.batchId}</span><span className="mt-1 block text-[10px] text-slate-500">{formatTaskTime(batch.createdAt)}</span></td>
                  <td className="px-2 py-3 text-center font-black text-violet-100">全模型一键测试</td>
                  <td className="px-2 py-3 text-center">全部</td>
                  <td className="px-2 py-2" title="聚合进度取所有子任务中的最小值">
                    <div className="history-progress-track mx-auto h-1.5 max-w-[110px] overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
                    <p className={cn('mt-1 text-center font-mono text-[10px] font-bold', tone.text)}>{progressPercent}% · {lifecycleStatusLabel(batch.status)}</p>
                  </td>
                  <td className="px-2 py-3 text-center font-mono">{elapsed !== null ? seconds(elapsed) : '—'}</td>
                  <td className="px-2 py-3 text-center font-mono">—</td>
                  <td className="px-2 py-3 text-center font-mono">—</td>
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
                  <td className="px-2 py-2 text-center font-mono text-cyan-200" title={item.asrSubId ?? rowId}><span className="block truncate">#{index + 1} {item.asrSubId ?? '历史结果'}{failed ? <span className="ml-2 rounded-full bg-rose-400/12 px-1.5 py-0.5 font-sans text-[10px] font-black text-rose-300">失败</span> : null}</span><span className="mt-1 block text-[10px] text-slate-500">{formatTaskTime(item.taskStatus?.createdAt ?? item.createdAt ?? evaluation?.createdAt)}</span></td>
                  <td className={cn('truncate px-2 py-3 text-center font-bold', failed ? 'text-rose-200' : 'text-slate-100')} title={evaluation?.model ?? item.request?.model}>{shortAsrModelName(evaluation?.model ?? item.request?.model)}</td>
                  <td className="px-2 py-3 text-center">{evaluation?.language ?? item.request?.language ?? '—'}</td>
                  <td className="px-2 py-2" title={item.taskStatus?.message ?? statusLabel}>
                    <div className="history-progress-track mx-auto h-1.5 max-w-[110px] overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
                    <p className={cn('mt-1 text-center font-mono text-[10px] font-bold', tone.text)}>{progressPercent}% · {statusLabel}</p>
                  </td>
                  <td className="px-2 py-3 text-center font-mono">{elapsedSec !== null ? seconds(elapsedSec) : '—'}</td>
                  <td className="px-2 py-3 text-center font-mono">{formatMetricValue(evaluation?.wer, 'percent')}</td>
                  <td className="px-2 py-3 text-center font-mono">{formatMetricValue(evaluation?.cer, 'percent')}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function CloneHistoryPanel({ history, batches, selectedCloneKey, onSelect, onOpenBatch, onOpenAsr, onOpenManual, onOpenFineTune, maxVisible }: { history: CloneHistoryEntry[]; batches: EvaluationBatch[]; selectedCloneKey?: string; onSelect: (cloneKey: string) => void; onOpenBatch: (batch: EvaluationBatch) => void; onOpenAsr: (asrSubId?: string) => void; onOpenManual: (item: CloneHistoryEntry) => void; onOpenFineTune: (item: CloneHistoryEntry) => void; maxVisible: number }) {
  if (!history.length && !batches.length) return null
  const rowCount = history.length + batches.length
  return (
    <section className="rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4">
      <div className="flex flex-col items-center justify-center gap-1 text-center">
        <SectionTitle>同一保护任务的克隆任务对比</SectionTitle>
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
            <tr><th className="px-2 py-2 text-center">任务</th><th className="px-2 py-2 text-center">克隆类型</th><th className="px-2 py-2 text-center">克隆模型</th><th className="px-2 py-2 text-center">标注来源</th><th className="px-2 py-2 text-center">进度</th><th className="px-2 py-2 text-center">处理时长</th><th className="px-2 py-2 text-center">参考标注</th><th className="px-2 py-2 text-center">原始相似度</th><th className="px-2 py-2 text-center">保护后相似度</th></tr>
          </thead>
          <tbody>
            {batches.map((batch) => {
              const progressPercent = Math.round(clamp(optionalNumber(batch.progress) ?? 0, 0, 1) * 100)
              const tone = progressTone(batch.status)
              const elapsed = batchElapsed(batch)
              return (
                <tr key={batch.batchId} role="button" tabIndex={0} onClick={() => onOpenBatch(batch)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onOpenBatch(batch) } }} className="cursor-pointer border-b border-violet-300/14 bg-violet-400/[0.06] hover:bg-violet-400/[0.11] focus:outline-none focus:ring-1 focus:ring-inset focus:ring-violet-300/40">
                  <td className="px-2 py-2 text-center font-mono text-violet-200" title={batch.batchId}><span className="block truncate">{batch.batchId}</span><span className="mt-1 block text-[10px] text-slate-500">{formatTaskTime(batch.createdAt)}</span></td>
                  <td className="px-2 py-3 text-center"><span className="rounded-full border border-violet-300/20 bg-violet-400/10 px-2 py-1 font-black text-violet-100">批次</span></td>
                  <td className="px-2 py-3 text-center font-black text-violet-100">全模型一键测试</td>
                  <td className="px-2 py-3 text-center">全部</td>
                  <td className="px-2 py-2" title="聚合进度取所有子任务中的最小值">
                    <div className="history-progress-track mx-auto h-1.5 max-w-[110px] overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
                    <p className={cn('mt-1 text-center font-mono text-[10px] font-bold', tone.text)}>{progressPercent}% · {lifecycleStatusLabel(batch.status)}</p>
                  </td>
                  <td className="px-2 py-3 text-center font-mono">{elapsed !== null ? seconds(elapsed) : '—'}</td>
                  <td className="px-2 py-3 text-center text-slate-400">完成 {batch.completedCount}/{batch.totalCount} · 失败 {batch.failedCount}</td>
                  <td className="px-2 py-3 text-center font-mono">—</td>
                  <td className="px-2 py-3 text-center font-mono">—</td>
                </tr>
              )
            })}
            {history.map((item, index) => {
              const request = item.request
              const asrAnnotation = request?.annotationSource === 'asr'
              const originalPrompt = request?.originalSpeakerPrompt ?? request?.speakerPrompt ?? ''
              const protectedPrompt = request?.protectedSpeakerPrompt ?? ''
              const lifecycleStatus = cloneHistoryLifecycleStatus(item)
              const failed = ['unavailable', 'failed', 'error', 'cancelled'].includes(lifecycleStatus)
              const progressPercent = Math.round(cloneHistoryProgress(item) * 100)
              const tone = progressTone(lifecycleStatus)
              const elapsed = optionalNumber(item.taskStatus?.elapsedSec)
              const openAnnotation = () => {
                if (!request) return
                if (asrAnnotation) onOpenAsr(request.annotationAsrSubId)
                else onOpenManual(item)
              }
              return (
                <tr key={item.key} role="button" tabIndex={0} title={failed ? cloneHistoryFailureReason(item) : item.taskStatus?.message ?? undefined} onClick={() => onSelect(item.key)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(item.key) } }} className={cn('cursor-pointer border-b border-cyan-300/8 last:border-0 hover:bg-cyan-300/[0.04] focus:outline-none focus:ring-1 focus:ring-inset focus:ring-cyan-300/35', selectedCloneKey === item.key && 'bg-cyan-300/[0.08]', failed && 'bg-rose-400/[0.035]')}>
                  <td className="px-2 py-2 text-center font-mono text-cyan-200" title={item.cloneSubId ?? item.cloneId ?? item.key}><span className="block truncate">#{index + 1} {item.cloneSubId ?? item.cloneId?.slice(0, 12) ?? '等待编号'}{failed ? <span className="ml-2 rounded-full bg-rose-400/12 px-1.5 py-0.5 font-sans text-[10px] font-black text-rose-300">失败</span> : null}</span><span className="mt-1 block text-[10px] text-slate-500">{formatTaskTime(item.taskStatus?.createdAt ?? item.createdAt)}</span></td>
                  <td className="px-2 py-3 text-center">
                    <span className="inline-flex items-center justify-center gap-1">
                      <span className="rounded-full border border-cyan-300/18 bg-cyan-400/8 px-2 py-1 font-black text-cyan-100">{request?.model ? cloneTypeLabel(request.model) : '—'}</span>
                      {item.result?.fineTune ? <button type="button" onClick={(event) => { event.stopPropagation(); onOpenFineTune(item) }} className="grid h-7 w-7 place-items-center rounded-[6px] border border-amber-300/18 text-amber-200 hover:bg-amber-300/10" aria-label="查看微调报告" title="查看微调报告"><Search className="h-3.5 w-3.5" /></button> : null}
                    </span>
                  </td>
                  <td className={cn('truncate px-2 py-3 text-center font-bold', failed ? 'text-rose-200' : 'text-slate-100')} title={request?.model}>{shortCloneModelName(request?.model)}</td>
                  <td className="px-2 py-3 text-center">
                    {request ? <button type="button" onClick={(event) => { event.stopPropagation(); openAnnotation() }} className={cn('rounded-full border px-2 py-1 font-bold underline-offset-2 hover:underline', asrAnnotation ? 'border-violet-300/20 bg-violet-400/10 text-violet-200' : 'manual-annotation-chip border-emerald-400/30 bg-emerald-500/15 text-emerald-300')}>{annotationSourceLabel(request.annotationSource)}</button> : '—'}
                  </td>
                  <td className="px-2 py-2" title={item.taskStatus?.message ?? lifecycleStatusLabel(lifecycleStatus)}>
                    <div className="history-progress-track mx-auto h-1.5 max-w-[110px] overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
                    <p className={cn('mt-1 text-center font-mono text-[10px] font-bold', tone.text)}>{progressPercent}% · {lifecycleStatusLabel(lifecycleStatus)}</p>
                  </td>
                  <td className="px-2 py-3 text-center font-mono">{elapsed !== null ? seconds(elapsed) : '—'}</td>
                  <td className="px-2 py-2">
                    {request ? <button type="button" onClick={(event) => { event.stopPropagation(); openAnnotation() }} className="block w-full rounded-[5px] px-1 py-1 text-left hover:bg-cyan-300/[0.05]">{asrAnnotation ? <span className="block space-y-0.5 leading-5"><span className="block truncate" title={originalPrompt}>原始：{originalPrompt || '—'}</span><span className="block truncate" title={protectedPrompt}>保护：{protectedPrompt || '—'}</span></span> : <span className="block truncate" title={originalPrompt}>{originalPrompt || '—'}</span>}</button> : <span className="block text-center text-slate-500">—</span>}
                  </td>
                  <td className="px-2 py-3 text-center font-mono">{formatCloneMetricNumber(item.result?.cloneEval?.originalSimilarity)}</td>
                  <td className="px-2 py-3 text-center font-mono">{formatCloneMetricNumber(item.result?.cloneEval?.protectedSimilarity)}</td>
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
            <p className="mt-1 font-mono text-xs text-slate-500">{item.cloneSubId ?? item.cloneId ?? item.key} · {request.model}</p>
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

function FineTuneReportModal({ item, onClose }: { item: CloneHistoryEntry | null; onClose: () => void }) {
  useEffect(() => {
    if (!item?.result?.fineTune) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [item, onClose])
  const report = item?.result?.fineTune
  if (!item || !report) return null
  const rows = [
    ['原始参考音频', report.original],
    ['保护参考音频', report.protected],
  ] as const
  return createPortal(
    <div className="fixed inset-0 z-[160] grid place-items-center bg-slate-950/80 px-4 py-8" role="dialog" aria-modal="true" aria-label="微调报告" onClick={onClose}>
      <div className="ui-card max-h-full w-full max-w-[760px] overflow-y-auto !bg-[#061426] p-5 shadow-[0_28px_90px_rgba(0,0,0,0.62)]" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-bold tracking-[0.12em] text-amber-300">现场微调报告</p>
            <h3 className="mt-2 text-xl font-black text-white">{shortCloneModelName(report.model ?? item.request?.model)}</h3>
            <p className="mt-1 font-mono text-xs text-slate-500">{item.cloneSubId ?? item.cloneId ?? item.key}{report.mode ? ` · ${report.mode}` : ''}</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-cyan-300/14 text-slate-300 hover:text-white" aria-label="关闭微调报告"><X className="h-4 w-4" /></button>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          {rows.map(([label, evidence]) => (
            <section key={label} className="rounded-[8px] border border-cyan-300/12 bg-slate-950 p-4">
              <p className="text-sm font-black text-cyan-100">{label}</p>
              <dl className="mt-3 grid grid-cols-[1fr_auto] gap-x-4 gap-y-2 text-xs">
                <dt className="text-slate-500">参考音频时长</dt><dd className="font-mono text-slate-200">{formatMetricValue(evidence?.sourceDurationSec, 'seconds')}</dd>
                <dt className="text-slate-500">训练音频时长</dt><dd className="font-mono text-slate-200">{formatMetricValue(evidence?.trainingDurationSec, 'seconds')}</dd>
                <dt className="text-slate-500">文本预处理</dt><dd className="font-mono text-slate-200">{formatMetricValue(evidence?.textSec, 'seconds')}</dd>
                <dt className="text-slate-500">HuBERT 特征</dt><dd className="font-mono text-slate-200">{formatMetricValue(evidence?.hubertSec, 'seconds')}</dd>
                <dt className="text-slate-500">语义特征</dt><dd className="font-mono text-slate-200">{formatMetricValue(evidence?.semanticSec, 'seconds')}</dd>
                <dt className="text-slate-500">GPT 训练</dt><dd className="font-mono text-slate-200">{formatMetricValue(evidence?.s1TrainSec, 'seconds')}</dd>
                <dt className="text-slate-500">SoVITS 训练</dt><dd className="font-mono text-slate-200">{formatMetricValue(evidence?.s2TrainSec, 'seconds')}</dd>
                <dt className="text-slate-500">推理耗时</dt><dd className="font-mono text-slate-200">{formatMetricValue(evidence?.inferenceWallSec, 'seconds')}</dd>
                <dt className="text-slate-500">单侧总耗时</dt><dd className="font-mono font-black text-amber-200">{formatMetricValue(evidence?.totalWallSec, 'seconds')}</dd>
              </dl>
            </section>
          ))}
        </div>
        <div className="mt-4 flex items-center justify-between rounded-[8px] border border-amber-300/14 bg-slate-950 px-4 py-3 text-sm">
          <span className="font-bold text-slate-400">原始与保护两侧合计耗时</span>
          <span className="font-mono font-black text-amber-200">{formatMetricValue(report.pairWallSec, 'seconds')}</span>
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
  const selectedModel = modelOptions.find((item) => item.value === form.model)
  const reusableAsrAnnotations = asrAnnotations
    .filter((item) => item.asrSubId && item.asr?.originalText?.trim() && item.asr?.protectedText?.trim())
    .filter((item) => {
      const query = annotationSearch.trim().toLowerCase()
      if (!query) return true
      return `${item.asr?.originalText ?? ''} ${item.asr?.protectedText ?? ''} ${item.asr?.model ?? ''} ${item.asrSubId ?? ''}`.toLowerCase().includes(query)
    })
    .slice()
    .reverse()

  const selectModel = (selected: BackendSelectOption) => {
    const languages = selected.languages?.length ? selected.languages : languageOptions
    onChange({
      ...form,
      model: selected.value,
      language: languages.includes(form.language ?? '') ? form.language : languages[0],
    })
  }

  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="语音克隆测试表单">
      <div className="ui-card max-h-[92vh] w-full max-w-[620px] overflow-y-auto !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-[20px] font-black text-white">语音克隆测试</h3>
            <p className="mt-1 text-xs text-slate-500">选择文本、模型和参考标注后开始测试</p>
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
                  {unavailable ? <span className="mt-0.5 block truncate text-[10px] font-medium text-amber-200">{item.status}</span> : null}
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
        {selectedModel?.promptRequired ? (
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
                      <p className="truncate text-xs font-black text-violet-100">{item.asr?.model ?? 'ASR'} · {item.createdAt ?? item.asrSubId}</p>
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

function BatchProgressModal({ batch, onClose }: { batch: EvaluationBatch | null; onClose: () => void }) {
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
            <p className="mt-1 font-mono text-xs text-slate-500">{batch.batchId}</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 text-slate-300 hover:text-white" aria-label="关闭"><X className="h-4 w-4" /></button>
        </div>
        <div className="mt-5 rounded-[9px] border border-violet-300/16 bg-violet-400/[0.07] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
            <span className="font-black text-violet-100">整体 {progressPercent}% · {lifecycleStatusLabel(batch.status)}</span>
            <span className="text-slate-400">完成 {batch.completedCount}/{batch.totalCount} · 失败 {batch.failedCount}</span>
          </div>
          <div className="history-progress-track mt-3 h-2 overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', tone.fill)} style={{ width: `${progressPercent}%` }} /></div>
          <p className="mt-2 text-xs text-slate-500">整体进度取全部预期子任务的最小有效进度；终态成功或失败均按 100% 参与计算。</p>
        </div>
        <div className="mt-4 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
          {batch.items.map((item) => {
            const itemProgress = clamp(optionalNumber(item.progress) ?? 0, 0, 1)
            const itemPercent = Math.round(itemProgress * 100)
            const itemTone = progressTone(item.status)
            const itemError = typeof item.error === 'string' ? item.error : item.error?.message
            const modelLabel = item.modelName || (batch.type === 'asr' ? shortAsrModelName(item.model) : shortCloneModelName(item.model))
            return (
              <div key={item.batchItemId} className={cn('rounded-[8px] border p-3', ['failed', 'error', 'cancelled'].includes(String(item.status).toLowerCase()) ? 'border-rose-300/18 bg-rose-400/[0.05]' : 'border-cyan-300/10 bg-slate-950/24')}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate font-black text-slate-100" title={item.model}>{modelLabel}</p>
                    <p className="mt-0.5 text-[11px] text-slate-500">{item.modelType || (batch.type === 'asr' ? 'ASR' : 'TTS 克隆')}{item.annotationSource ? ` · ${annotationSourceLabel(item.annotationSource)}` : ''}</p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className={cn('font-mono text-xs font-black', itemTone.text)}>{itemPercent}% · {lifecycleStatusLabel(item.status)}</p>
                    <p className="mt-0.5 font-mono text-[10px] text-slate-500">{optionalNumber(item.elapsedSec) !== null ? seconds(optionalNumber(item.elapsedSec) as number) : '—'}</p>
                  </div>
                </div>
                <div className="history-progress-track mt-2 h-1.5 overflow-hidden rounded-full bg-slate-800"><div className={cn('h-full rounded-full transition-all duration-300', itemTone.fill)} style={{ width: `${itemPercent}%` }} /></div>
                <p className={cn('mt-2 truncate text-xs', itemError ? 'text-rose-300' : 'text-slate-400')} title={itemError || item.message || undefined}>{itemError || item.message || '等待后端更新任务阶段'}</p>
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

function ScoreBox({ label, value, red, compact, foot }: { label: ReactNode; value: string; red?: boolean; compact?: boolean; foot?: ReactNode }) {
  return (
    <div className={cn('relative rounded-[9px] border border-cyan-300/12 bg-slate-950/16 text-center', compact ? 'p-2.5' : 'p-3')}>
      <div className={cn('relative', compact ? 'min-h-9' : 'min-h-8')}>
        <p className={cn('absolute inset-x-0 top-1/2 line-clamp-2 -translate-y-1/2 break-words px-8 text-center font-black text-slate-300', compact ? 'text-[12px] leading-4' : 'text-[13px] leading-5')}>{label}</p>
        {foot ? <div className="absolute right-0 top-1/2 -translate-y-1/2"><MetricInfoButton title={label}>{foot}</MetricInfoButton></div> : null}
      </div>
      <div className="mt-2 grid justify-items-center">
        <span className={cn(compact ? 'text-[19px]' : 'text-[24px]', 'break-words font-black leading-none', red ? 'text-red-300' : 'text-cyan-300')}>
          {value}
        </span>
      </div>
    </div>
  )
}

function QualityPanel({ result, embedded }: { result: TaskResult; embedded?: boolean }) {
  const snr = optionalNumber(result.protectionQuality?.snr) ?? optionalNumber(result.quality.snr)
  const pesq = optionalNumber(result.protectionQuality?.pesq) ?? optionalNumber(result.quality.pesq)
  const stoi = optionalNumber(result.protectionQuality?.stoi)
  const backendMos = optionalNumber(result.protectionQuality?.mos)
  const manualMosKey = `manual-mos:${result.taskId || 'current'}`
  const readManualMos = (key: string) => {
    const saved = window.localStorage.getItem(key)
    if (saved === null) return null
    const value = Number(saved)
    return Number.isFinite(value) ? clamp(value, 1, 5) : null
  }
  const [manualMosState, setManualMosState] = useState<{ key: string; value: number | null }>(() => ({ key: manualMosKey, value: readManualMos(manualMosKey) }))
  const manualMos = manualMosState.key === manualMosKey ? manualMosState.value : readManualMos(manualMosKey)
  const setManualMos = (value: number | null) => setManualMosState({ key: manualMosKey, value })
  const [editingMos, setEditingMos] = useState(false)
  const [mosDraft, setMosDraft] = useState('')
  const mos = backendMos ?? manualMos
  const missingReasons = [
    pesq === null ? ['PESQ', metricReason(result, ['protectionQuality.pesq'])] : null,
    stoi === null ? ['STOI', metricReason(result, ['protectionQuality.stoi'])] : null,
  ].filter((item): item is [string, string] => Boolean(item?.[1]))

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
        <QualityMetric label="SNR" value={formatMetricValue(snr, 'db')} tag={snr === null ? '未生成' : '信噪比'} tone="green" foot="原始语音与扰动噪声的强弱比；数值越高，音频越接近原音。" />
        <QualityMetric label="PESQ" value={formatMetricValue(pesq, 'number')} tag={pesq === null ? '未生成' : '听感质量'} tone="blue" foot="模拟人耳评价语音质量，分数越高表示听感越好。" />
        <QualityMetric label="STOI" value={formatMetricValue(stoi, 'number')} tag={stoi === null ? '未生成' : '可懂度'} tone="blue" foot="衡量语音是否容易听懂，越接近 1 表示可懂度越高。" />
        <QualityMetric
          label="人工MOS"
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
          foot="人工试听给出的主观质量评分，范围为 1–5 分；分数越高表示听感越好。"
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
      const message = '指定时间帧心理声学曲线加载失败，请稍后重试。'
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
            <SectionTitle>心理声学阈值分析</SectionTitle>
            <MetricInfoButton title="心理声学阈值分析">{modeDescription}</MetricInfoButton>
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

      {timeDialogOpen ? (
        <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="选择心理声学分析时间点">
          <form
            className="ui-card w-full max-w-[440px] !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]"
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
        <div className="absolute left-1/2 top-11 z-20 w-[180px] -translate-x-1/2 rounded-[8px] border border-cyan-300/18 bg-slate-950 p-1 shadow-[0_18px_45px_rgba(0,0,0,0.42)]" role="menu">
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

function QualityMetric({ label, value, tag, tone, onClick, title, foot }: { label: string; value: ReactNode; tag: string; tone: 'green' | 'blue' | 'orange'; onClick?: () => void; title?: string; foot?: ReactNode }) {
  return (
    <div
      className={cn('relative rounded-[9px] border border-cyan-300/12 bg-slate-950/16 px-3 py-3.5 text-center', onClick && 'cursor-pointer transition hover:border-orange-300/28 hover:bg-orange-300/[0.04]')}
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
      <div className="relative min-h-7">
        <p className="absolute inset-x-0 top-1/2 -translate-y-1/2 truncate px-8 text-center text-[12px] font-black text-slate-300">{label}</p>
        {foot ? <div className="absolute right-0 top-1/2 -translate-y-1/2"><MetricInfoButton title={label}>{foot}</MetricInfoButton></div> : null}
      </div>
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
  { key: 'L2', formula: 'L_2', label: '扰动能量约束', description: '扰动范数约束', colorClass: 'bg-violet-300' },
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
                暂未记录逐步优化损失，当前仅可在详细数据中查看最终 loss。
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
            </div>
          ))}
          <div className="rounded-[7px] border border-cyan-300/12 bg-slate-950/20 px-5 py-5">
            <div className="flex items-center justify-between gap-4">
              <p className="text-[11px] font-bold text-slate-400">平均每次迭代耗时</p>
              <p className="text-[14px] font-black text-cyan-200">{avgIterationSec === null ? '未生成' : `${avgIterationSec.toFixed(2)} s / step`}</p>
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
              <MathText formula={loss.formula} className="align-[-1px]" />：暂未生成
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
    ['保护任务', result.taskId],
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
      <div className="ui-card w-full max-w-[560px] !bg-[#061426] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div>
            <h3 className="text-[20px] font-black text-white">任务信息</h3>
            <p className="mt-1 text-xs text-slate-500">查看本次保护任务的完整配置与状态</p>
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

  const runDownload = async () => {
    try {
      const file = await downloadProtectedAudio(result.taskId)
      downloadBlob(file.blob, file.filename)
      pushToast({ kind: 'success', title: '下载已开始', description: file.filename })
    } catch (error) {
      pushToast({ kind: 'error', title: '导出暂不可用', description: error instanceof Error ? error.message : '请稍后重试。' })
    }
  }

  return (
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
    </div>
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
  if (/cannot import name ['"]pipeline['"].*transformers/i.test(value)) return 'Wav2Vec2 运行环境初始化失败；该条历史任务没有产生转写文本，请重新测试。'
  if (/out of memory|cuda.*memory/i.test(value)) return '显存不足，模型未能完成转写，请稍后重新测试。'
  if (/missing|not found|no such file/i.test(value)) return '模型文件不完整或不可用，请检查模型部署后重新测试。'
  return value.split('\n')[0]
}

function formatRatioPercent(value: unknown, options?: { clampToUnit?: boolean }) {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  const normalized = options?.clampToUnit ? clamp(numberValue, 0, 1) : numberValue
  return `${(normalized * 100).toFixed(2)}%`
}

function formatRadarScore(value: unknown) {
  const numberValue = optionalNumber(value)
  if (numberValue === null) return '未生成'
  return `${clamp(numberValue, 0, 100).toFixed(2)} 分`
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
  asrHistory?: AsrEvalResponse[]
  cloneHistory?: CloneVoiceResult[]
}

function averageAvailable(values: Array<number | null>) {
  const available = values.filter((value): value is number => value !== null && Number.isFinite(value))
  return available.length ? available.reduce((sum, value) => sum + value, 0) / available.length : null
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
    return '调参建议（ASR 联动）：暂未生成对应 ASR 任务，请先点击右上角“ASR 测试”完成识别评估。'
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
    return '调参建议（ASR 平均联动）：现有 ASR 任务均未返回可用 WER/CER，暂不据此调整 \\(\\lambda_{\\mathrm{sem}}\\)。'
  }

  const metricText = formatRatioPercent(metric)
  const sampleText = `基于 ${metrics.length} 次可用 ASR 结果，平均 ${metricName} 为 ${metricText}`
  if (metric < asrWeakDisruptionThreshold) {
    return `调参建议（ASR 平均联动）：${sampleText}，整体语义干扰较弱；建议提高 \\(\\lambda_{\\mathrm{sem}}\\) 后重新保护并复测。`
  }
  if (metric < asrStrongDisruptionThreshold) {
    return `调参建议（ASR 平均联动）：${sampleText}，整体语义干扰已有一定效果；若优先阻断语义链路，可小幅提高 \\(\\lambda_{\\mathrm{sem}}\\)。`
  }
  return `调参建议（ASR 平均联动）：${sampleText}，整体语义干扰效果较明显，当前可保持 \\(\\lambda_{\\mathrm{sem}}\\)。`
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
    return '调参建议（克隆联动）：暂未生成对应克隆任务，请先点击右上角“语音克隆测试”完成声音身份评估。'
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
    return '调参建议（克隆平均联动）：现有克隆任务均未返回保护后声纹相似度，暂不据此调整 \\(\\lambda_{\\mathrm{id}}\\)。'
  }

  const similarityText = protectedSimilarity.toFixed(2)
  const sampleText = `基于 ${validEvaluations.length} 次可用克隆结果，保护后声纹相似度平均值为 ${similarityText}`
  const dropText = similarityDropRate === null ? '' : `，相对原始克隆的平均下降幅度为 ${formatRatioPercent(similarityDropRate)}`
  if (originalSimilarity !== null && originalSimilarity < speakerSameIdentityThreshold) {
    return `调参建议（克隆平均联动）：基于 ${validEvaluations.length} 次可用克隆结果，原始克隆声纹相似度平均值为 ${originalSimilarity.toFixed(2)}，整体克隆基线偏弱；建议先更换克隆模型或样本复测，暂不据此调整 \\(\\lambda_{\\mathrm{id}}\\)。`
  }
  if (protectedSimilarity >= speakerHighSimilarityThreshold) {
    return `调参建议（克隆平均联动）：${sampleText}${dropText}，整体声音身份残留较高；建议提高 \\(\\lambda_{\\mathrm{id}}\\) 后重新保护并复测。`
  }
  if (protectedSimilarity >= speakerSameIdentityThreshold) {
    return `调参建议（克隆平均联动）：${sampleText}${dropText}，整体仍有一定声音身份特征残留；建议小幅提高 \\(\\lambda_{\\mathrm{id}}\\)。`
  }
  return `调参建议（克隆平均联动）：${sampleText}${dropText}，整体声音身份相似度已明显降低，当前可保持 \\(\\lambda_{\\mathrm{id}}\\)。`
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
    qualityNotes.push(`STOI 为 ${stoi.toFixed(2)}，${level}。`)
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

function generateAsrInsights(asrEval: AsrEval, editStats: EditMetrics | null) {
  const wer = optionalNumber(asrEval.wer) ?? (editStats?.level === 'word' ? editStats.werOrCer : null)
  const cer = optionalNumber(asrEval.cer) ?? (editStats?.level === 'char' ? editStats.werOrCer : null)
  const insertRate = optionalNumber(asrEval.insertRate) ?? editStats?.insertRate ?? null
  const items: string[] = []
  if ((wer ?? 0) >= 0.3 || (cer ?? 0) >= 0.3) items.push('WER/CER 较高，ASR 识别受到干扰。')
  if ((insertRate ?? 0) >= 0.2) items.push('插入率较高，句子结构稳定性下降。')
  if (items.length === 0) items.push('ASR 文本级错误率较低或指标不足；任务级 Token 变化率与语义漂移在共享语义指标区域单独展示。')
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
    return <div className="grid h-full place-items-center text-xs text-slate-500">暂未生成心理声学频谱数据</div>
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
