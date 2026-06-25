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
import type { AsrMetrics, CapabilitiesResponse, CloneVoiceRequest, CloneVoiceResult, LossFinal, LossTrendPoint, ProtectionRuntimeConfig, TaskResult, TaskStatusResponse } from '@/types/task'
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
  const displayData = asrOverride ? { ...data, asr: asrOverride, asrModel: asrOverride.model ?? data.asrModel } : data

  return (
    <div className="-mx-5 max-w-none space-y-5 pb-6">
      <SummaryBar result={displayData} />

      <div className="grid grid-cols-[minmax(0,1fr)_380px] items-stretch gap-5 max-xl:grid-cols-1">
        <AudioCompare result={displayData} onAsrUpdated={(asr) => setAsrOverrideState({ taskId, asr })} />
        <Aside result={displayData} />
      </div>

      <div className="grid min-h-[380px] grid-cols-[minmax(0,1fr)_minmax(520px,0.9fr)] items-stretch gap-6 max-xl:grid-cols-1">
        <TrendPanel result={displayData} />
        <InterpretationPanel result={displayData} />
      </div>
    </div>
  )
}

function SummaryBar({ result }: { result: TaskResult }) {
  const score = optionalNumber(result.score)
  const isEffective = score !== null && score >= 80
  const verdict = result.verdict || (score === null ? '未生成' : isEffective ? '防护有效' : '防护无效')

  return (
    <section className="ui-card grid h-[74px] grid-cols-[250px_180px_250px_170px_230px_minmax(260px,1fr)] items-center px-5 max-2xl:grid-cols-[1.05fr_0.78fr_1.08fr_0.75fr_1fr_1.42fr] max-xl:h-auto max-xl:grid-cols-3 max-xl:gap-y-4 max-xl:py-4">
      <SummaryItem icon={<ClipboardList />} label="任务 ID" value={result.taskId} copy />
      <SummaryItem icon={<ShieldCheck />} label="任务状态" value={statusText[result.status] ?? result.status} green={result.status === 'completed' || result.status === 'success'} />
      <SummaryItem icon={<Clock3 />} label="完成时间" value={result.completedAt ?? '-'} />
      <SummaryItem icon={<Clock3 />} label="处理耗时" value={typeof result.elapsedSec === 'number' ? formatElapsed(result.elapsedSec) : '-'} />
      <SummaryItem icon={<Sparkles />} label="防护模式" value={modeText[result.mode] ?? result.mode} green />
      <div className="flex h-full items-center justify-end gap-4 border-l border-cyan-300/10 pl-5 whitespace-nowrap">
        <span className="text-xs text-slate-500">综合判定</span>
        <ShieldCheck className={cn('h-11 w-11', isEffective ? 'text-emerald-300' : 'text-red-300')} />
        <div className="text-left">
          <p className={cn('text-[27px] font-black leading-none', isEffective ? 'text-emerald-300' : 'text-red-300')}>{verdict}</p>
          <p className="mt-1 text-xs text-slate-400">{isEffective ? '满足当前安全性阈值' : '未达到当前安全性阈值'}</p>
        </div>
        <div className="grid h-[58px] w-[58px] place-items-center rounded-full border-4 border-dashed border-emerald-400/70 text-center text-sm font-black text-emerald-300">
          {formatOptionalNumber(score, 1)}
        </div>
      </div>
    </section>
  )
}

function formatElapsed(seconds: number) {
  const hh = Math.floor(seconds / 3600)
  const mm = Math.floor((seconds % 3600) / 60)
  const ss = seconds % 60
  return [hh, mm, ss].map((value) => String(value).padStart(2, '0')).join(':')
}

function SummaryItem({ icon, label, value, green, copy }: { icon: ReactNode; label: string; value: string; green?: boolean; copy?: boolean }) {
  return (
    <div className="flex min-w-0 items-center justify-center gap-3 border-r border-cyan-300/10 px-4 whitespace-nowrap">
      <span className="text-slate-500 [&_svg]:h-4 [&_svg]:w-4">{icon}</span>
      <div className="min-w-0">
        <p className="text-xs text-slate-500">{label}</p>
        <p className={cn('mt-1 truncate text-[14px] font-bold text-slate-200', green && 'text-emerald-300')}>{value}</p>
      </div>
      {copy ? <Copy className="h-4 w-4 shrink-0 text-slate-500" /> : null}
    </div>
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
  const activeAsr = result.asr
  const originalText = activeAsr.originalText ?? ''
  const protectedText = activeAsr.protectedText ?? ''
  const asrEditStats = getTextEditStats(originalText, protectedText)
  const asrMetrics = [
    ['WER（词错率）', formatOptionalPercent(activeAsr.wer)],
    ['CER（字错率）', formatOptionalPercent(activeAsr.cer ?? asrEditStats.cer)],
    ['Token 错误率', formatOptionalPercent(activeAsr.tokenErrorRate ?? activeAsr.tokenChangeRate)],
    ['SD（语义漂移）', formatOptionalPercent(activeAsr.semanticDrift)],
    ['IR（插入率）', formatOptionalPercent(activeAsr.insertRate ?? asrEditStats.insertRate)],
    ['DR（删除率）', formatOptionalPercent(activeAsr.deleteRate ?? asrEditStats.deleteRate)],
  ]

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
      const response = await runAsrEval(result.taskId, { model: asrModel, referenceText: originalText || undefined })
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
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <SectionTitle info>结果对比</SectionTitle>
          <div className="flex items-center gap-2">
            {[
              ['protect', '保护'],
              ['asr', 'ASR'],
              ['clone', '克隆'],
            ].map(([key, label]) => (
              <button key={key} type="button" onClick={() => setActivePanel(key as ComparePanel)} className={cn('h-9 rounded-[7px] border border-cyan-300/14 px-3 text-sm font-black text-slate-300 transition hover:text-white', activePanel === key && 'bg-cyan-400/14 text-cyan-200')} title={`查看${label}结果`}>
                {label}
              </button>
            ))}
          </div>
        </div>
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
          <div className="grid items-center gap-6 pr-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
            <AudioCard title="原始音频（未保护）" audio={originalAudio} color="#00aef0" />
            <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-cyan-300/28 bg-slate-950/70 text-[18px] font-black text-white shadow-[0_0_24px_rgba(56,189,248,0.12)]">VS</div>
            <AudioCard title="保护音频（已防护）" audio={protectedAudio} color="#22c55e" green onPlayRequest={loadProtectedAudio} />
          </div>
        ) : null}
        {activePanel === 'asr' ? (
          <div className="px-1">
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_288px_minmax(0,1fr)]">
              <TextBox title="原始转写（ASR）" text={originalText || '未生成'} foot="原始音频经 ASR 识别得到的参考转写文本" />
              <div className="grid grid-cols-2 content-center gap-3">
                {asrMetrics.map(([label, value]) => (
                  <ScoreBox key={label} label={label} value={value} red compact />
                ))}
              </div>
              <TextBox title="保护音频转写（ASR）" text={protectedText || '未生成'} foot="红色为新增内容，绿色删除线为原文缺失内容" content={originalText && protectedText ? buildTextDiff(originalText, protectedText) : undefined} />
            </div>
          </div>
        ) : null}
        {activePanel === 'clone' ? (
          <div className="grid items-center gap-6 pl-1 lg:grid-cols-[minmax(0,1fr)_58px_minmax(0,1fr)]">
            {cloneLoading ? (
              <>
                <LoadingCard title="克隆原语音" progress={cloneTaskStatus?.stage === 'downstream_tts_eval' ? cloneTaskStatus.progress : undefined} message={cloneTaskStatus?.stage === 'downstream_tts_eval' ? cloneTaskStatus.message : undefined} />
                <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
                <LoadingCard title="克隆保护语音" progress={cloneTaskStatus?.stage === 'downstream_tts_eval' ? cloneTaskStatus.progress : undefined} message={cloneTaskStatus?.stage === 'downstream_tts_eval' ? cloneTaskStatus.message : undefined} />
              </>
            ) : cloneResult ? (
              <>
                <AudioCard title="克隆原语音" audio={cloneResult.originalCloneAudio} color="#a78bfa" />
                <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
                <AudioCard title="克隆保护语音" audio={cloneResult.protectedCloneAudio} color="#f59e0b" />
              </>
            ) : (
              <>
                <CloneEmptyCard title="克隆原语音" />
                <div className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-violet-300/28 bg-slate-950/70 text-[18px] font-black text-white">VS</div>
                <CloneEmptyCard title="克隆保护语音" />
              </>
            )}
          </div>
        ) : null}
      </div>
      <div className="mt-5 grid grid-cols-[1.08fr_0.92fr] items-stretch gap-5 max-lg:grid-cols-1">
        <SpeakerPanel result={result} embedded />
        <QualityPanel result={result} embedded />
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

function CloneEmptyCard({ title }: { title: string }) {
  return (
    <div className="grid h-[252px] place-items-center rounded-[9px] border border-violet-300/18 bg-slate-950/18 p-5 text-center">
      <div>
        <TestTube2 className="mx-auto h-9 w-9 text-violet-200" />
        <p className="mt-4 text-sm font-black text-slate-100">{title}</p>
        <p className="mt-2 text-xs text-slate-400">需要先填写克隆测试表单。</p>
      </div>
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

function getTextEditStats(original: string, next: string) {
  const a = Array.from(original)
  const b = Array.from(next)
  if (a.length === 0) {
    return {
      cer: undefined,
      insertRate: undefined,
      deleteRate: undefined,
    }
  }

  const dp = Array.from({ length: a.length + 1 }, () => Array<number>(b.length + 1).fill(0))

  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }

  let i = 0
  let j = 0
  let insertions = 0
  let deletions = 0

  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      i += 1
      j += 1
    } else if (j < b.length && (i === a.length || dp[i][j + 1] >= dp[i + 1]?.[j])) {
      insertions += 1
      j += 1
    } else if (i < a.length) {
      deletions += 1
      i += 1
    }
  }

  const base = a.length

  return {
    cer: (insertions + deletions) / base,
    insertRate: insertions / base,
    deleteRate: deletions / base,
  }
}

function buildTextDiff(original: string, next: string) {
  const a = Array.from(original)
  const b = Array.from(next)
  const dp = Array.from({ length: a.length + 1 }, () => Array<number>(b.length + 1).fill(0))

  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }

  const nodes: ReactNode[] = []
  let i = 0
  let j = 0
  let key = 0

  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      nodes.push(a[i])
      i += 1
      j += 1
    } else if (j < b.length && (i === a.length || dp[i][j + 1] >= dp[i + 1]?.[j])) {
      nodes.push(
        <span key={`ins-${key}`} className="text-red-300">
          {b[j]}
        </span>,
      )
      j += 1
      key += 1
    } else if (i < a.length) {
      nodes.push(
        <span key={`del-${key}`} className="text-emerald-300 line-through decoration-emerald-300/70">
          {a[i]}
        </span>,
      )
      i += 1
      key += 1
    }
  }

  return nodes
}

function ScoreBox({ label, value, red, compact }: { label: string; value: string; red?: boolean; compact?: boolean }) {
  return (
    <div className={cn('rounded-[9px] border border-cyan-300/12 bg-slate-950/16 text-center', compact ? 'h-[64px] p-2.5' : 'h-[82px] p-3')}>
      <p className="whitespace-nowrap text-[11px] leading-4 text-slate-400">{label}</p>
      <div className="mt-2 grid justify-items-center">
        <span className={cn(compact ? 'text-[19px]' : 'text-[24px]', 'font-black leading-none', red ? 'text-red-300' : 'text-cyan-300')}>
          {value}
        </span>
      </div>
    </div>
  )
}

function SpeakerPanel({ result, embedded }: { result: TaskResult; embedded?: boolean }) {
  const simBefore = optionalNumber(result.speaker.simBefore)
  const simAfter = optionalNumber(result.speaker.simAfter)
  const embeddingBefore = optionalNumber(result.speaker.embeddingDistanceBefore)
  const embeddingAfter = optionalNumber(result.speaker.embeddingDistanceAfter)
  const simDropRate = optionalNumber(result.speaker.simDropRate) ?? relativeDrop(simBefore, simAfter)
  const embeddingIncrease = relativeIncrease(embeddingBefore, embeddingAfter)

  return (
    <section className={cn('flex flex-col overflow-x-auto', embedded ? 'rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4' : 'ui-card p-5')}>
      <div className={cn('border-b border-cyan-300/10 pb-3', embedded ? 'mb-4' : 'mb-5')}>
        <SectionTitle>Feature / 声学特征分析</SectionTitle>
      </div>

      <div className="mx-auto grid w-full max-w-[540px] flex-1 grid-cols-2 items-center gap-5">
        <FeatureStatCard
          title="Feature 相似度（越低越好）"
          before={formatOptionalNumber(simBefore, 3)}
          after={formatOptionalNumber(simAfter, 3)}
          delta={`↓ ${formatPercent(simDropRate)}`}
          foot="计算方法：x-vector 余弦相似度"
          tone="green"
        />

        <FeatureStatCard
          title="Embedding 距离（越大越好）"
          before={formatOptionalNumber(embeddingBefore, 3)}
          after={formatOptionalNumber(embeddingAfter, 3)}
          delta={`↑ ${formatPercent(embeddingIncrease)}`}
          foot="计算方法：ECAPA-TDNN 向量距离"
          tone="red"
        />
      </div>
    </section>
  )
}

function FeatureStatCard({ title, before, after, delta, foot, tone }: { title: string; before: string; after: string; delta: string; foot: string; tone: 'green' | 'red' }) {
  return (
    <div className="h-[168px] rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-4">
      <h3 className="whitespace-nowrap text-[13px] font-bold leading-5 text-slate-300">{title}</h3>
      <div className="mt-3 grid grid-cols-[1fr_28px_1fr] items-center text-center text-[20px]">
        <span className="text-slate-200">{before}</span>
        <span className="text-slate-400">→</span>
        <span className="text-emerald-300">{after}</span>
      </div>
      <div className={cn('mt-2 rounded-[5px] py-2 text-center font-black', tone === 'green' ? 'bg-emerald-400/14 text-emerald-300' : 'bg-red-400/12 text-red-300')}>{delta}</div>
      <p className="mt-2 truncate text-[11px] leading-4 text-slate-500">{foot}</p>
    </div>
  )
}

function QualityPanel({ result, embedded }: { result: TaskResult; embedded?: boolean }) {
  return (
    <section className={cn('h-full overflow-hidden', embedded ? 'rounded-[9px] border border-cyan-300/12 bg-slate-950/12 p-4' : 'ui-card p-5')}>
      <SectionTitle>感知质量评估</SectionTitle>
      <div className="mt-5 grid grid-cols-3 gap-3">
        <QualityMetric label="SNR（信噪比）" value={formatOptionalNumber(result.quality.snr, 1, ' dB')} tag={result.quality.snr === null ? '未生成' : 'computed'} tone="green" />
        <QualityMetric label="PESQ" value={formatOptionalNumber(result.quality.pesq, 2)} tag={result.quality.pesq === null ? '未生成' : 'perception'} tone="blue" />
        <QualityMetric label="听感保真（MOS-LQO）" value={formatOptionalNumber(result.quality.mosLqo, 2, ' / 5')} tag={result.quality.mosLqo === null ? '未生成' : 'perception'} tone="orange" />
      </div>
      <div className="mt-5 h-[126px] overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/16 px-4 py-3">
        <div className="mb-1 flex items-center justify-between">
          <h3 className="whitespace-nowrap text-[13px] font-bold text-slate-300">心理声学阈值分析（关键频段）</h3>
          <div className="flex gap-4 text-[10px] text-slate-400">
            <span className="text-emerald-300">— 掩蔽阈值</span>
            <span className="text-lime-300">— 防护扰动谱</span>
          </div>
        </div>
        <LineChart result={result} />
      </div>
    </section>
  )
}

function QualityMetric({ label, value, tag, tone }: { label: string; value: string; tag: string; tone: 'green' | 'blue' | 'orange' }) {
  return (
    <div className="h-[86px] rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-3 text-center">
      <p className="whitespace-nowrap text-[11px] text-slate-400">{label}</p>
      <p className={cn('mt-1 text-[20px] font-black leading-none', tone === 'green' && 'text-emerald-300', tone === 'blue' && 'text-cyan-300', tone === 'orange' && 'text-orange-300')}>{value}</p>
      <span className={cn('mt-1.5 inline-block rounded px-3 py-0.5 text-[11px] font-bold', tone === 'green' && 'bg-emerald-400/14 text-emerald-300', tone === 'blue' && 'bg-cyan-400/14 text-cyan-300', tone === 'orange' && 'bg-orange-400/14 text-orange-300')}>{tag}</span>
    </div>
  )
}

const lossDefinitions: Array<{ key: keyof Pick<LossTrendPoint, 'Lfeat' | 'Lsem' | 'Lpsy' | 'L2'>; formula: string; altFormula?: string; label: string; description: string; colorClass: string }> = [
  { key: 'Lfeat', formula: 'L_{\\mathrm{feat}}', label: 'Feature Loss', description: '特征 / 音色损失', colorClass: 'bg-cyan-300' },
  { key: 'Lsem', formula: 'L_{\\mathrm{sem}}', label: 'Semantic Loss', description: '语义损失', colorClass: 'bg-emerald-300' },
  { key: 'Lpsy', formula: 'L_{\\mathrm{psy}}', label: 'Psychoacoustic Loss', description: '心理声学损失，量级可能较小', colorClass: 'bg-amber-300' },
  { key: 'L2', formula: 'L_2', altFormula: '\\lVert\\delta\\rVert_2', label: 'L2 Constraint', description: '扰动范数约束', colorClass: 'bg-violet-300' },
]

function TrendPanel({ result }: { result: TaskResult }) {
  const trend = result.charts.optimizationTrend
  const lossFinal = result.generation?.lossFinal ?? finalLossFromTrend(trend)
  const missingLosses = lossDefinitions.filter((loss) => trend.length > 0 && trend.every((point) => point[loss.key] === null || point[loss.key] === undefined))
  const steps = result.generation?.steps ?? lastStep(trend)
  const avgIterationSec = typeof result.elapsedSec === 'number' && steps && steps > 0 ? result.elapsedSec / steps : null

  return (
    <section className="ui-card flex h-full min-h-[380px] flex-col overflow-hidden p-7">
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
                <p className="text-[13px] font-black text-white">{formatLossNumber(lossFinal?.[loss.key])}</p>
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

function InterpretationPanel({ result }: { result: TaskResult }) {
  const simBefore = optionalNumber(result.speaker.simBefore)
  const simAfter = optionalNumber(result.speaker.simAfter)
  const simDropRate = optionalNumber(result.speaker.simDropRate) ?? relativeDrop(simBefore, simAfter)
  const score = optionalNumber(result.score)
  const verdict = result.verdict || (score === null ? '未生成' : score >= 80 ? '防护有效' : '防护无效')
  const editStats = getTextEditStats(result.asr.originalText ?? '', result.asr.protectedText ?? '')
  const cer = result.asr.cer ?? editStats.cer
  const tokenErrorRate = result.asr.tokenErrorRate ?? result.asr.tokenChangeRate
  const items = [
    `语义层面：WER ${formatOptionalPercent(result.asr.wer)}，CER ${formatOptionalPercent(cer)}，Token 错误率 ${formatOptionalPercent(tokenErrorRate)}。`,
    `Feature 层面：相似度从 ${formatOptionalNumber(simBefore, 3)} 降至 ${formatOptionalNumber(simAfter, 3)}（↓${formatPercent(simDropRate)}）。`,
    `听感层面：PESQ=${formatOptionalNumber(result.quality.pesq, 2)}，MOS-LQO=${formatOptionalNumber(result.quality.mosLqo, 2)}；缺失项按未生成展示，不做推断。`,
    `综合结论：综合评分 ${formatOptionalNumber(score, 1)}，判定为「${verdict}」。`,
  ]

  return (
    <section className="ui-card flex h-full min-h-[380px] flex-col overflow-hidden p-7">
      <SectionTitle>
        结果解读 <span className="text-sm font-normal text-slate-500">（自动生成）</span>
      </SectionTitle>
      <div className="mt-6 grid flex-1 grid-cols-1 content-start gap-4 rounded-[7px] border border-cyan-300/10 bg-slate-950/12 p-5 text-[14px] leading-7 text-slate-200">
        {items.map((item) => (
          <p key={item} className="flex min-w-0 gap-3">
            <CheckCircle2 className="mt-1.5 h-4 w-4 shrink-0 text-emerald-300" />
            <span>{item}</span>
          </p>
        ))}
      </div>
      <p className="mt-2 text-right text-[11px] text-slate-500">以上分析基于系统自动评估，仅供评审参考。</p>
    </section>
  )
}

function Aside({ result }: { result: TaskResult }) {
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)
  const taskInfo = [
    ['提交时间', result.submittedAt ?? result.createdAt ?? result.originalAudio.uploadedAt ?? '-'],
    ['输入来源', result.inputSource ?? '手动上传'],
    ['音频时长', formatDurationSeconds(getAudioDuration(result.originalAudio))],
    ['语言类型', result.language ?? '未标注'],
    ['处理模型', result.processingModel ?? result.asrModel ?? modeText[result.mode] ?? result.mode],
    ['优化目标', result.optimizationTarget ?? result.mode],
  ]

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
    <aside className="grid h-full grid-rows-2 gap-5 max-xl:grid-rows-none">
      <div className="ui-card px-5 py-6">
        <h2 className="mb-5 text-[17px] font-black text-white">任务信息</h2>
        {taskInfo.map(([label, value]) => (
          <p key={label} className="mb-3 grid grid-cols-[78px_1fr] text-[13px] leading-5">
            <span className="text-slate-500">{label}</span>
            <span className="truncate font-semibold text-slate-300">{value}</span>
          </p>
        ))}
        <p className="mt-2 grid grid-cols-[78px_1fr] text-[13px] leading-5">
          <span className="text-slate-500">参数配置</span>
          <button type="button" onClick={() => pushToast({ kind: 'info', title: '参数配置', description: '可通过任务详情接口查看完整参数：GET /api/tasks/{taskId}/details。' })} className="text-left font-bold text-cyan-300">查看详情 ›</button>
        </p>
      </div>

      <div className="ui-card p-5">
        <h2 className="mb-5 text-[18px] font-black text-white">操作与导出</h2>
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
    </aside>
  )
}

function formatFileSize(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)}MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${bytes}B`
}

function formatPercent(value: number) {
  return `${(value <= 1 ? value * 100 : value).toFixed(1)}%`
}

function formatOptionalPercent(value: unknown) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numberValue)) return '无'
  return `${(numberValue <= 1 ? numberValue * 100 : numberValue).toFixed(1)}%`
}

function optionalNumber(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function formatOptionalNumber(value: unknown, digits = 2, suffix = '') {
  const numberValue = optionalNumber(value)
  return numberValue === null ? '未生成' : `${numberValue.toFixed(digits)}${suffix}`
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
    Lfeat: last.Lfeat,
    Lsem: last.Lsem,
    Lpsy: last.Lpsy,
    L2: last.L2,
    total: last.total,
  }
}

function lastStep(points: LossTrendPoint[]) {
  const step = optionalNumber(points.at(-1)?.step)
  return step && step > 0 ? step : null
}

function relativeDrop(before: number | null, after: number | null) {
  if (before === null || after === null || !Number.isFinite(before) || before === 0) return 0
  return Math.max(0, (before - after) / Math.abs(before))
}

function relativeIncrease(before: number | null, after: number | null) {
  if (before === null || after === null || !Number.isFinite(before) || before === 0) return 0
  return Math.max(0, (after - before) / Math.abs(before))
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

function LineChart({ result }: { result: TaskResult }) {
  const points = result.charts.psychoacoustic
  const width = 430
  const height = 58
  if (points.length === 0) {
    return <div className="grid h-full place-items-center text-xs text-slate-500">后端未生成该图表数据</div>
  }
  const values = points.flatMap((p) => [p.maskingThreshold, p.perturbation].filter((value): value is number => typeof value === 'number' && Number.isFinite(value)))
  const max = Math.max(...values, 1)
  const min = Math.min(...values, 0)
  const span = Math.max(1, max - min)
  const toPoints = (key: 'maskingThreshold' | 'perturbation') =>
    points
      .map((point, index) => {
        const x = (index / Math.max(1, points.length - 1)) * width
        const y = height - 6 - (((point[key] ?? 0) - min) / span) * (height - 12)
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })
      .join(' ')

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[58px] w-full overflow-hidden">
      {[10, 28, 46].map((y) => (
        <line key={y} x1="0" x2={width} y1={y} y2={y} stroke="rgba(148,163,184,.13)" />
      ))}
      <polyline points={toPoints('maskingThreshold')} fill="none" stroke="#22c55e" strokeWidth="2" />
      <polyline points={toPoints('perturbation')} fill="none" stroke="#86efac" strokeDasharray="6 5" strokeWidth="2" />
      {points.filter((_, index) => index % 4 === 0).map((point, index) => (
        <text key={point.frequency} x={index * 4 * (width / Math.max(1, points.length - 1))} y="56" fontSize="9" fill="#64748b">
          {point.frequency >= 1000 ? `${Math.round(point.frequency / 1000)}k` : point.frequency}
        </text>
      ))}
    </svg>
  )
}
