import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import {
  CheckSquare,
  ChevronDown,
  ChevronUp,
  Info,
  Loader2,
  Mic,
  Settings,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  StopCircle,
  UploadCloud,
  Waves,
  X,
} from 'lucide-react'

import { createProtectionTask, getTaskStatus, uploadFile } from '@/services/apiClient'
import { useCapabilitiesQuery } from '@/hooks/useCapabilitiesQuery'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { CapabilitiesResponse, NumericConfigRange, ProtectionMode, ProtectionRuntimeConfig, ProtectionTarget, ProtectionTaskRequest, RuntimeModelOption, TaskStatusResponse } from '@/types/task'
import { cn } from '@/lib/utils'
import { AudioPlayer } from '@/components/audio/AudioPlayer'
import { MathText } from '@/components/common/MathText'
import { ModelInformationModal } from '@/components/common/ModelInformationModal'
import { WorkspaceEvaluationPanel } from '@/components/evaluation/WorkspaceEvaluationPanel'
import { formatTaskFailure } from '@/utils/apiError'
import { formatDurationSeconds, getAudioDuration, getRecordedExtension, getRecorderMimeType, readAudioDuration } from '@/utils/audio'
import { formatBytes, shortHash } from '@/utils/format'

const TASK_STATUS_POLL_MS = 1000

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

function configFromCapabilities(capabilities: CapabilitiesResponse | null | undefined): ProtectionRuntimeConfig | undefined {
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

type UiModelOption = RuntimeModelOption & { label: string }

function optionItems(options?: Array<string | RuntimeModelOption>): UiModelOption[] {
  return (options ?? []).map((option) => (typeof option === 'string' ? { label: option, name: option, value: option } : { ...option, label: option.label ?? option.value }))
}

function optionValues(options?: Array<string | RuntimeModelOption>) {
  return (options ?? []).map((option) => (typeof option === 'string' ? option : option.value)).filter(Boolean)
}

function pctFromRange(value: number, range: NumericConfigRange) {
  if (range.max <= range.min) return 0
  return ((value - range.min) / (range.max - range.min)) * 100
}

function clampToRange(value: number, range: NumericConfigRange) {
  return Math.max(range.min, Math.min(range.max, value))
}

function numericFromPreset(value: unknown, fallback: number, range: NumericConfigRange) {
  return clampToRange(typeof value === 'number' && Number.isFinite(value) ? value : fallback, range)
}

function decimalPlacesFromStep(step: number) {
  if (!Number.isFinite(step) || step <= 0) return 3
  const normalized = step.toString().toLowerCase()
  if (normalized.includes('e-')) return Number(normalized.split('e-')[1])
  const decimal = normalized.split('.')[1]
  return decimal ? decimal.length : 0
}

function formatParameterValue(value: number, range: NumericConfigRange) {
  if (!Number.isFinite(value)) return ''
  const decimals = Math.min(8, Math.max(0, decimalPlacesFromStep(range.step)))
  const fixed = value.toFixed(decimals)
  return fixed.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')
}

function formatScientificParameterValue(value: number) {
  if (!Number.isFinite(value)) return ''
  if (value === 0) return '0'
  return value.toExponential(2).replace(/\.?0+e/, 'e').replace('e-0', 'e-').replace('e+0', 'e+')
}

function isDoneStatus(status: TaskStatusResponse) {
  return status.status === 'completed' || status.status === 'success'
}

function isFailedStatus(status: TaskStatusResponse) {
  return status.status === 'failed' || status.status === 'error'
}

function isTransientTaskNotFound(error: unknown) {
  const message = error instanceof Error ? error.message : ''
  return /task not found/i.test(message) || /TASK_NOT_FOUND/.test(message)
}

function revokeObjectUrl(url?: string) {
  if (url?.startsWith('blob:')) URL.revokeObjectURL(url)
}

function splitUploadedAt(value?: string) {
  const normalized = String(value ?? '').trim().replace('T', ' ')
  const match = normalized.match(/^(.+?)[,\s]+(\d{1,2}:\d{2}(?::\d{2})?)/)
  return match ? { date: match[1], time: match[2] } : { date: normalized || '-', time: '-' }
}

export function WorkspacePage() {
  const navigate = useNavigate()
  const mountedRef = useRef(true)
  const [running, setRunning] = useState(false)
  const [, setTaskId] = useState<string>()
  const activeTaskTokenRef = useRef(0)
  const { data: capabilities, error: capabilitiesQueryError } = useCapabilitiesQuery()

  const pushToast = useAppStore((state) => state.pushToast)
  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const setCurrentTaskStatus = useTaskStore((state) => state.setCurrentTaskStatus)
  const runtimeConfig = configFromCapabilities(capabilities)
  const configError = capabilitiesQueryError
    ? capabilitiesQueryError instanceof Error ? capabilitiesQueryError.message : '无法读取系统参数配置。'
    : capabilities && !runtimeConfig ? '系统暂未返回运行参数配置。' : null

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const applyStatus = (status: TaskStatusResponse, taskToken: number) => {
    if (!mountedRef.current || taskToken !== activeTaskTokenRef.current) return
    setCurrentTaskStatus(status)
  }

  const waitForBackendTask = async (createdTaskId: string, taskToken: number) => {
    const startedAt = Date.now()
    while (mountedRef.current) {
      let latest: TaskStatusResponse
      try {
        latest = await getTaskStatus(createdTaskId)
      } catch (error) {
        if (Date.now() - startedAt < 5000 && isTransientTaskNotFound(error)) {
          await wait(250)
          continue
        }
        throw error
      }
      applyStatus(latest, taskToken)
      if (isDoneStatus(latest) || isFailedStatus(latest)) return latest
      await wait(TASK_STATUS_POLL_MS)
    }
    return null
  }

  const startTask = async (payload: ProtectionTaskRequest) => {
    if (!uploadedFile?.fileId) {
      pushToast({ kind: 'error', title: '无法创建任务', description: '请先上传音频并等待文件准备完成。' })
      return
    }

    const taskToken = activeTaskTokenRef.current + 1
    activeTaskTokenRef.current = taskToken
    try {
      setRunning(true)

      const created = await createProtectionTask({ ...payload, fileId: uploadedFile?.fileId ?? payload.fileId })
      setTaskId(created.taskId)
      pushToast({ kind: 'success', title: '保护任务已创建', description: '正在生成保护音频。' })

      const finalStatus = await waitForBackendTask(created.taskId, taskToken)
      if (!finalStatus) return
      if (taskToken !== activeTaskTokenRef.current) return
      if (isFailedStatus(finalStatus)) {
        setRunning(false)
        pushToast({ kind: 'error', title: '任务执行失败', description: formatTaskFailure(finalStatus) })
        return
      }
      setRunning(false)
      navigate(`/results/${created.taskId}`)
    } catch (error) {
      if (taskToken === activeTaskTokenRef.current) setRunning(false)
      pushToast({ kind: 'error', title: '任务创建失败', description: error instanceof Error ? error.message : '请稍后重试。' })
    }
  }

  return (
    <div className="workspace-page-shell grid min-h-0 grid-rows-[minmax(0,1fr)]">
      <div className="workspace-grid grid min-h-0 grid-cols-[minmax(360px,0.9fr)_minmax(390px,1fr)_minmax(420px,1.05fr)] gap-3 max-xl:grid-cols-1">
        <AudioAccessCard maxAudioSizeBytes={runtimeConfig?.constraints?.maxAudioSizeBytes} />
        <StrategyConfigCard running={running} runtimeConfig={runtimeConfig} configError={configError} averageStepSec={capabilities?.runtimePerformance?.averageStepSec} onStart={(payload) => void startTask(payload)} />
        <WorkspaceEvaluationPanel runtimeConfig={runtimeConfig} modelTypes={runtimeConfig?.modelTypes ?? capabilities?.modelTypes} />
      </div>
    </div>
  )
}

function isAlignedToStep(value: number, min: number, step: number) {
  if (!Number.isFinite(step) || step <= 0) return true
  const quotient = (value - min) / step
  const nearest = Math.round(quotient)
  return Math.abs(quotient - nearest) <= 1e-7 * Math.max(1, Math.abs(quotient))
}

function AudioAccessCard({ maxAudioSizeBytes }: { maxAudioSizeBytes?: number }) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const recordTickRef = useRef<number | null>(null)
  const [inputMode, setInputMode] = useState<'upload' | 'record'>('upload')
  const [recording, setRecording] = useState(false)
  const [recordingSec, setRecordingSec] = useState(0)
  const pushToast = useAppStore((state) => state.pushToast)
  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const setUploadedFile = useTaskStore((state) => state.setUploadedFile)

  const clearRecordingTimer = () => {
    if (recordTickRef.current) {
      window.clearTimeout(recordTickRef.current)
      recordTickRef.current = null
    }
  }

  const scheduleRecordingTick = () => {
    recordTickRef.current = window.setTimeout(() => {
      setRecordingSec((value) => value + 1)
      scheduleRecordingTick()
    }, 1000)
  }

  const releaseRecorderStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    recorderRef.current = null
  }

  useEffect(() => {
    return () => {
      clearRecordingTimer()
      releaseRecorderStream()
    }
  }, [])

  const handleFile = async (file?: File) => {
    if (!file) return
    if (maxAudioSizeBytes && file.size > maxAudioSizeBytes) {
      pushToast({ kind: 'error', title: '文件过大', description: `单文件大小不能超过 ${formatBytes(maxAudioSizeBytes)}。` })
      return
    }

    const objectUrl = URL.createObjectURL(file)
    const format = file.name.split('.').pop()?.toUpperCase() ?? file.type.split('/').pop()?.toUpperCase() ?? 'AUDIO'
    const baseMeta = {
      filename: file.name,
      sizeBytes: file.size,
      format,
      objectUrl,
      rawFile: file,
      uploadedAt: new Date().toLocaleString('zh-CN', { hour12: false }),
      fingerprint: shortHash(`${file.name}-${file.size}-${file.lastModified}`),
    }

    revokeObjectUrl(uploadedFile?.objectUrl)
    setUploadedFile(baseMeta)

    try {
      const durationSec = await readAudioDuration(objectUrl).catch(() => undefined)
      const localMeta = durationSec ? { ...baseMeta, durationSec, duration: durationSec } : baseMeta
      setUploadedFile(localMeta)

      const backendMeta = await uploadFile(file)
      setUploadedFile({ ...localMeta, ...backendMeta, objectUrl, rawFile: file })
      pushToast({ kind: 'success', title: '音频已上传', description: `${file.name} 已准备完成。` })
    } catch (error) {
      revokeObjectUrl(objectUrl)
      setUploadedFile(null)
      pushToast({ kind: 'error', title: '上传失败', description: error instanceof Error ? error.message : '请重新选择音频文件。' })
    }
  }

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      pushToast({ kind: 'error', title: '浏览器不支持录音', description: '请使用支持 MediaRecorder 的现代浏览器，并在 localhost 或 HTTPS 环境下测试。' })
      return
    }

    try {
      setRecordingSec(0)
      chunksRef.current = []

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mimeType = getRecorderMimeType()
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      recorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }

      recorder.onstop = () => {
        const type = recorder.mimeType || mimeType || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type })
        const extension = getRecordedExtension(type)
        const recordId = typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `${Date.now()}_${Math.random().toString(16).slice(2)}`
        const name = `record_${recordId}.${extension}`
        const file = new File([blob], name, { type, lastModified: Date.now() })

        clearRecordingTimer()
        releaseRecorderStream()
        setRecording(false)
        void handleFile(file)
      }

      recorder.start()
      setRecording(true)
      scheduleRecordingTick()
      pushToast({ kind: 'success', title: '录音已开始', description: '请对着麦克风朗读测试语音。' })
    } catch (error) {
      clearRecordingTimer()
      releaseRecorderStream()
      setRecording(false)
      pushToast({ kind: 'error', title: '无法启动录音', description: error instanceof Error ? error.message : '请检查麦克风权限。' })
    }
  }

  const stopRecording = () => {
    if (!recorderRef.current || recorderRef.current.state === 'inactive') return
    clearRecordingTimer()
    recorderRef.current.stop()
    pushToast({ kind: 'info', title: '录音已结束', description: '录音文件正在接入。' })
  }

  const toggleRecording = () => {
    if (recording) {
      stopRecording()
      return
    }
    void startRecording()
  }

  const displayFile = uploadedFile ?? {
    filename: '尚未选择音频',
    sizeBytes: 0,
    format: '-',
  }
  const duration = getAudioDuration(uploadedFile)
  const maxAudioSizeLabel = maxAudioSizeBytes ? formatBytes(maxAudioSizeBytes) : '系统限制'
  const uploadMoment = splitUploadedAt(uploadedFile?.uploadedAt)

  return (
    <section className="workspace-audio-card workspace-hover-surface ui-card h-full min-h-0 p-4">
      <h2 className="flex items-center gap-2 text-[21px] font-black text-white">
        音频接入
        <Info className="h-4 w-4 text-slate-500" />
      </h2>
      <div className="mt-4 grid grid-cols-2 text-center text-sm font-bold">
        <button
          type="button"
          onClick={() => setInputMode('upload')}
          className={cn('border-b pb-3 transition', inputMode === 'upload' ? 'border-cyan-400 text-cyan-300' : 'border-slate-700 text-slate-500 hover:text-slate-300')}
        >
          上传音频
        </button>
        <button
          type="button"
          onClick={() => setInputMode('record')}
          className={cn('border-b pb-3 transition', inputMode === 'record' ? 'border-cyan-400 text-cyan-300' : 'border-slate-700 text-slate-500 hover:text-slate-300')}
        >
          录音输入
        </button>
      </div>

      {inputMode === 'upload' ? (
        <div
          className="workspace-audio-dropzone workspace-hover-card mt-5 grid place-items-center rounded-[12px] border border-dashed border-cyan-300/55 bg-sky-400/5 text-center transition hover:border-cyan-200 hover:bg-cyan-400/10"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault()
            void handleFile(event.dataTransfer.files[0])
          }}
        >
          <div className="workspace-audio-dropzone-content flex w-full flex-col items-center px-6 pt-6 pb-6 text-center">
            <UploadCloud className="mb-4 h-14 w-14 text-cyan-300" />
            <p className="text-[18px] font-black text-white">拖拽音频文件到此处，或点击上传</p>
            <p className="mt-3 text-[15px] leading-6 text-slate-300">
              支持 .wav / .flac / .mp3 / .m4a / .webm
              <br />
              单文件 ≤ {maxAudioSizeLabel}
            </p>
            <input ref={inputRef} type="file" accept=".wav,.mp3,.flac,.m4a,.webm,audio/*" className="hidden" onChange={(event) => void handleFile(event.target.files?.[0])} />
            <span onClick={() => inputRef.current?.click()} className="workspace-hover-control mt-3 inline-flex h-11 min-w-[180px] items-center justify-center gap-2 rounded-full bg-cyan-300 px-5 text-[17px] font-medium text-slate-950 shadow-[0_0_28px_rgba(34,211,238,0.25)] cursor-pointer">
              <UploadCloud className="h-5 w-5" />
              选择文件
            </span>
          </div>
        </div>
      ) : (
        <div className="workspace-audio-dropzone workspace-hover-card mt-5 grid place-items-center rounded-[12px] border border-dashed border-cyan-300/55 bg-sky-400/5 text-center transition hover:border-cyan-200 hover:bg-cyan-400/10">
          <div className="workspace-audio-dropzone-content workspace-recorder-content flex w-full flex-col items-center px-6 pt-6 pb-6 text-center">
            <button
              type="button"
              onClick={toggleRecording}
              className={cn(
                'workspace-recorder-control workspace-hover-control',
                'grid h-[4.5rem] w-[4.5rem] place-items-center rounded-full border-2 transition',
                recording ? 'border-red-300 bg-red-400/18 text-red-200 shadow-[0_0_24px_rgba(248,113,113,0.35)]' : 'border-cyan-300/40 bg-cyan-400/12 text-cyan-200',
              )}
              aria-pressed={recording}
            >
              {recording ? <StopCircle className="h-8 w-8" /> : <Mic className="h-8 w-8" />}
            </button>
            <p className="workspace-recorder-title mt-4 text-[18px] font-black text-white">{recording ? '点击停止录音' : '点击开始录音'}</p>
            <p className="workspace-recording-time mt-2 font-mono text-[20px] font-black text-cyan-100">{formatDurationSeconds(recordingSec)}</p>
            <TinyWave color={recording ? '#f87171' : '#00aef0'} className="workspace-recorder-wave mx-auto mt-5 h-9 w-full max-w-[330px]" />
          </div>
        </div>
      )}

      <div className="mt-9 border-t border-cyan-300/10 pt-3">
        <h3 className="text-[15px] font-bold text-slate-300">已上传文件</h3>
      </div>
      <div className="uploaded-audio-panel workspace-hover-card mt-2 rounded-[8px] border border-cyan-300/12 bg-[#07192d]/85 p-3">
        <div className="uploaded-audio-header mb-2 flex min-w-0 items-center gap-3">
          <FilenameWithExtension filename={displayFile.filename} className="min-w-0 flex-1 font-bold text-white" />
          <span className={cn('shrink-0 rounded-full px-3 py-1 text-xs font-bold', uploadedFile ? 'bg-emerald-400/12 text-emerald-300' : 'bg-slate-500/12 text-slate-400')}>
            {uploadedFile ? '就绪' : '待上传'}
          </span>
        </div>
        <AudioPlayer
          src={uploadedFile?.objectUrl ?? uploadedFile?.audioUrl}
          title="播放预览"
          filename={uploadedFile?.filename}
          disabledReason="请选择或拖拽音频文件后预览"
        />
        <div className="uploaded-audio-meta mt-3 grid grid-cols-3 gap-x-3 gap-y-4 border-t border-cyan-300/10 pt-3 text-sm leading-5">
          {[
            ['文件名', displayFile.filename],
            ['时长', duration ? `${duration.toFixed(2)}s` : '待解析'],
            ['采样率', uploadedFile?.sampleRate ? `${uploadedFile.sampleRate.toLocaleString()} Hz` : '待解析'],
            ['声道', uploadedFile?.channels ? `${uploadedFile.channels}` : '待解析'],
            ['大小', uploadedFile ? formatBytes(uploadedFile.sizeBytes) : '-'],
            ['格式', displayFile.format],
            ['上传日期', uploadMoment.date],
            ['上传时间', uploadMoment.time],
          ].map(([label, value]) => (
            <div key={label} className="min-w-0">
              <p className="text-slate-500">{label}</p>
              {label === '文件名' ? (
                <FilenameWithExtension filename={String(value)} className="mt-1 font-semibold text-slate-200" />
              ) : (
                <p className="mt-1 truncate font-semibold text-slate-200">{value}</p>
              )}
            </div>
          ))}
          <div className="uploaded-audio-fingerprint order-last min-w-0">
            <p className="text-slate-500">文件指纹</p>
            <p className="mt-1 truncate font-mono font-bold text-slate-200" title={uploadedFile?.fingerprint ?? '-'}>{uploadedFile?.fingerprint ?? '-'}</p>
          </div>
        </div>
      </div>

    </section>
  )
}

function FilenameWithExtension({ filename, className }: { filename: string; className?: string }) {
  const dotIndex = filename.lastIndexOf('.')
  const hasExtension = dotIndex > 0 && dotIndex < filename.length - 1
  const stem = hasExtension ? filename.slice(0, dotIndex) : filename
  const extension = hasExtension ? filename.slice(dotIndex) : ''

  return (
    <span className={cn('flex min-w-0 items-baseline', className)} title={filename}>
      <span className="min-w-0 truncate">{stem}</span>
      {extension ? <span className="shrink-0">{extension}</span> : null}
    </span>
  )
}

function StrategyConfigCard({
  running,
  runtimeConfig,
  configError,
  averageStepSec,
  onStart,
}: {
  running: boolean
  runtimeConfig?: ProtectionRuntimeConfig
  configError?: string | null
  averageStepSec?: number | null
  onStart: (payload: ProtectionTaskRequest) => void
}) {
  const pushToast = useAppStore((state) => state.pushToast)
  const [selectedMode, setSelectedMode] = useState<Exclude<ProtectionMode, 'joint'>>('standard')
  const [selectedTarget, setSelectedTarget] = useState<ProtectionTarget | 'joint'>('joint')
  const [lambdaModalOpen, setLambdaModalOpen] = useState(false)
  const [epsilon, setEpsilon] = useState(0)
  const [steps, setSteps] = useState(0)
  const [lambdaSem, setLambdaSem] = useState(0)
  const [lambdaFeat, setLambdaFeat] = useState(0)
  const [lambdaPsy, setLambdaPsy] = useState(0)
  const [lambdaL2, setLambdaL2] = useState(0)
  const [selectedSemanticEncoders, setSelectedSemanticEncoders] = useState<string[]>([])
  const [selectedFeatureModels, setSelectedFeatureModels] = useState<string[]>([])
  const [configurationChangedSinceSubmission, setConfigurationChangedSinceSubmission] = useState(false)
  const [modelModalOpen, setModelModalOpen] = useState(false)

  const applyPreset = (mode: Exclude<ProtectionMode, 'joint'>, config: ProtectionRuntimeConfig) => {
    const defaults = config.defaults
    const ranges = config.ranges
    const identityRange = ranges.weightIdentity ?? ranges.weightFeature
    const preset = config.modePresets?.[mode]
    setEpsilon(numericFromPreset(preset?.optimization?.epsilon, defaults.optimization.epsilon, ranges.epsilon))
    setSteps(Math.max(1, Math.round(numericFromPreset(preset?.optimization?.steps, defaults.optimization.steps, ranges.steps))))
    setLambdaSem(numericFromPreset(preset?.semantic?.weightSemantic, defaults.semantic.weightSemantic ?? ranges.weightSemantic.min, ranges.weightSemantic))
    setLambdaFeat(numericFromPreset(preset?.timbre?.weightIdentity ?? preset?.timbre?.weightFeature, defaults.timbre.weightIdentity ?? defaults.timbre.weightFeature ?? identityRange.min, identityRange))
    setLambdaPsy(numericFromPreset(preset?.psychoacoustic?.weightPsy, defaults.psychoacoustic.weightPsy ?? ranges.weightPsy.min, ranges.weightPsy))
    setLambdaL2(numericFromPreset(preset?.optimization?.weightL2 ?? defaults.optimization.weightL2, defaults.optimization.weightL2 ?? ranges.weightL2.min, ranges.weightL2))
  }

  useEffect(() => {
    if (!runtimeConfig) return
    const timeoutId = window.setTimeout(() => {
      const defaults = runtimeConfig.defaults
      const semanticOptions = optionValues(runtimeConfig.formSchema?.modelOptions?.semanticEncoders ?? runtimeConfig.models.semantic)
      const featureOptions = optionValues(runtimeConfig.formSchema?.modelOptions?.timbreEncoders ?? runtimeConfig.models.timbre ?? runtimeConfig.models.feature)
      const semanticEnabled = defaults.semantic.enabled || defaults.targets.includes('semantic')
      const timbreEnabled = defaults.timbre.enabled || defaults.targets.includes('timbre')
      const defaultMode = defaults.mode

      setSelectedMode(defaultMode)
      setSelectedTarget(defaultMode === 'custom' ? 'joint' : semanticEnabled && timbreEnabled ? 'joint' : semanticEnabled ? 'semantic' : timbreEnabled ? 'timbre' : 'joint')
      applyPreset(defaultMode, runtimeConfig)
      setSelectedSemanticEncoders(defaults.semantic.encoders?.length ? defaults.semantic.encoders : semanticOptions)
      setSelectedFeatureModels(defaults.timbre.encoders.length ? defaults.timbre.encoders : featureOptions)
    }, 0)
    return () => window.clearTimeout(timeoutId)
  }, [runtimeConfig])

  if (!runtimeConfig) {
    return (
      <section className="workspace-strategy-card workspace-hover-surface ui-card flex h-full min-h-0 flex-col p-5">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-[21px] font-black text-white">
            防护策略配置
            <Info className="h-4 w-4 text-slate-500" />
          </h2>
          <span className="rounded-[6px] border border-cyan-300/14 bg-white/[0.03] px-3 py-2 text-sm text-slate-300">
            参数状态
          </span>
        </div>
        <div className="workspace-config-loading workspace-hover-card grid min-h-0 flex-1 place-items-center rounded-[7px] border border-cyan-300/14 bg-slate-950/20 p-6 text-center">
          <div>
            {configError ? <Info className="mx-auto h-10 w-10 text-amber-300" /> : <Loader2 className="mx-auto h-10 w-10 animate-spin text-cyan-300" />}
            <p className="mt-4 text-base font-black text-white">{configError ? '系统参数读取失败' : '正在读取系统参数'}</p>
            <p className="mt-2 text-sm leading-6 text-slate-400">{configError || '暂时无法读取默认参数和模型列表，请稍后重试。'}</p>
          </div>
        </div>
        <div className="mt-5 grid gap-3">
          <button disabled className="cyan-button inline-flex h-[50px] items-center justify-center gap-2 rounded-[7px] text-[16px] font-black opacity-60">
            <ShieldCheck className="h-5 w-5" />
            开始生成保护音频
          </button>
        </div>
      </section>
    )
  }

  const epsilonRange = runtimeConfig.ranges.epsilon
  const stepsRange = runtimeConfig.ranges.steps
  const lambdaSemRange = runtimeConfig.ranges.weightSemantic
  const lambdaFeatRange = runtimeConfig.ranges.weightIdentity ?? runtimeConfig.ranges.weightFeature
  const lambdaPsyRange = runtimeConfig.ranges.weightPsy
  const lambdaL2Range = runtimeConfig.ranges.weightL2
  const semanticOptionItems = optionItems(runtimeConfig.formSchema?.modelOptions?.semanticEncoders ?? runtimeConfig.models.semantic)
  const featureOptionItems = optionItems(runtimeConfig.formSchema?.modelOptions?.timbreEncoders ?? runtimeConfig.models.timbre ?? runtimeConfig.models.feature)
  const semanticOptions = semanticOptionItems.map((option) => option.value)
  const featureOptions = featureOptionItems.map((option) => option.value)
  const modeOptions = runtimeConfig.modes ?? [
    { value: 'standard', label: '标准保护', description: '平衡安全与听感', targetPolicy: 'selectable' },
    { value: 'strong', label: '强保护', description: '更强安全性，略降听感', targetPolicy: 'selectable' },
    { value: 'high_fidelity', label: '高保真', description: '更优听感，安全性适中', targetPolicy: 'selectable' },
    { value: 'custom', label: '自定义', description: '自由调整参数，仅允许联合', targetPolicy: 'joint_only' },
  ]
  const targetOptions = runtimeConfig.targets ?? [
    { value: 'semantic', label: '语义防护', description: '降低 ASR/LLM 理解概率' },
    { value: 'timbre', label: '声音身份防护', description: '阻断声学特征重建' },
    { value: 'joint', label: '联合防护', description: '语义 + 声音身份联合防护' },
  ]
  const targetIconMap = {
    semantic: { Icon: ShieldCheck, tone: 'green' },
    timbre: { Icon: Waves, tone: 'blue' },
    joint: { Icon: ShieldCheck, tone: 'cyan' },
  } as const
  const targetLocked = selectedMode === 'custom'
  const measuredAverageStepSec = typeof averageStepSec === 'number' && Number.isFinite(averageStepSec) && averageStepSec > 0
    ? averageStepSec
    : null

  const handleModeChange = (mode: string) => {
    setConfigurationChangedSinceSubmission(true)
    const nextMode = mode as Exclude<ProtectionMode, 'joint'>
    setSelectedMode(nextMode)
    if (nextMode === 'custom') setSelectedTarget('joint')
    applyPreset(nextMode, runtimeConfig)
  }

  const switchToCustom = () => {
    setConfigurationChangedSinceSubmission(true)
    if (selectedMode !== 'custom') {
      setSelectedMode('custom')
      setSelectedTarget('joint')
      pushToast({ id: 'workspace-custom-mode-switch', kind: 'info', title: '已切换自定义', description: '参数配置仅在自定义联合模式生效。', dedupeMs: 2500 })
    }
  }

  const updateNumber = (setter: (value: number) => void) => (value: number) => {
    switchToCustom()
    setter(value)
  }

  const toggleOption = (value: string, selected: string[], setter: (value: string[]) => void) => {
    switchToCustom()
    const next = selected.includes(value) ? selected.filter((item) => item !== value) : [...selected, value]
    setter(next.length ? next : [value])
  }

  const openModelModal = () => {
    switchToCustom()
    setModelModalOpen(true)
  }

  const handleTargetChange = (target: ProtectionTarget | 'joint') => {
    setConfigurationChangedSinceSubmission(true)
    setSelectedTarget(target)
  }

  const buildPayload = (): ProtectionTaskRequest => {
    const semanticEnabled = selectedTarget === 'semantic' || selectedTarget === 'joint'
    const featureEnabled = selectedTarget === 'timbre' || selectedTarget === 'joint'
    const safeSteps = Math.max(1, Math.round(clampToRange(steps, stepsRange)))
    const safeSemanticEncoders = selectedSemanticEncoders.length ? selectedSemanticEncoders : semanticOptions
    const safeFeatureModels = selectedFeatureModels.length ? selectedFeatureModels : featureOptions
    return {
      mode: selectedMode,
      profile: runtimeConfig.activeDefaultProfile ?? 'formal',
      targets: [semanticEnabled ? 'semantic' : null, featureEnabled ? 'timbre' : null].filter(Boolean) as Array<'semantic' | 'timbre'>,
      semantic: {
        enabled: semanticEnabled,
        encoders: safeSemanticEncoders,
        tokenizerPath: runtimeConfig.defaults.semantic.tokenizerPath,
        hubertPath: runtimeConfig.defaults.semantic.hubertPath,
        whisperPath: runtimeConfig.defaults.semantic.whisperPath,
        weightSemantic: clampToRange(lambdaSem, lambdaSemRange),
      },
      timbre: {
        enabled: featureEnabled,
        mode: 'untargeted',
        encoders: safeFeatureModels,
        weightIdentity: clampToRange(lambdaFeat, lambdaFeatRange),
        weightFeature: clampToRange(lambdaFeat, lambdaFeatRange),
      },
      psychoacoustic: { enabled: true, weightPsy: clampToRange(lambdaPsy, lambdaPsyRange) },
      optimization: { epsilon: clampToRange(epsilon, epsilonRange), steps: safeSteps, weightL2: clampToRange(lambdaL2, lambdaL2Range) },
    }
  }

  return (
    <section className="workspace-strategy-card workspace-hover-surface ui-card flex h-full min-h-0 flex-col overflow-hidden p-5">
      <div className="workspace-strategy-header mb-6 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-[21px] font-black text-white">
          防护策略配置
          <Info className="h-4 w-4 text-slate-500" />
        </h2>
        <button
          type="button"
          onClick={() => pushToast({ kind: 'info', title: '策略模板', description: '当前页面已内置标准、强保护、高保真与高级自定义模板。' })}
          className="workspace-hover-control rounded-[6px] border border-cyan-300/14 bg-white/[0.03] px-3 py-2 text-sm text-slate-300"
        >
          <Settings className="mr-1 inline h-4 w-4" />
          策略模板
        </button>
      </div>

      <div className="workspace-strategy-content min-h-0 flex-1">
      <ConfigBlock title="保护模式">
        <div className="grid grid-cols-4 gap-2.5">
          {modeOptions.map((mode) => (
            <button
              key={mode.value}
              type="button"
              onClick={() => handleModeChange(mode.value)}
              className={cn(
                'workspace-mode-option workspace-hover-control h-[70px] rounded-[6px] border bg-slate-950/18 px-3 text-left',
                mode.value === selectedMode ? 'border-cyan-400 bg-cyan-400/8 text-cyan-200' : 'border-cyan-300/12 text-slate-300',
              )}
            >
              <p className="font-black">{mode.label}</p>
              <p className="mt-1 text-xs text-slate-500">{mode.description}</p>
            </button>
          ))}
        </div>
      </ConfigBlock>

      <ConfigBlock title="防护目标" helper={targetLocked ? '固定联合' : '三模式任选'}>
        <div className="grid grid-cols-3 gap-2.5">
          {targetOptions.map((target) => {
            const targetValue = target.value as ProtectionTarget | 'joint'
            const selected = selectedTarget === targetValue
            const { Icon, tone } = targetIconMap[targetValue] ?? targetIconMap.joint
            const disabled = targetLocked && targetValue !== 'joint'
            return (
            <button
              key={target.value}
              type="button"
              disabled={disabled}
              onClick={() => handleTargetChange(targetValue)}
              className={cn(
                'workspace-target-option workspace-hover-control h-[74px] rounded-[6px] border px-3 py-2 text-left',
                selected ? 'border-cyan-400 bg-cyan-400/10' : 'border-cyan-300/12 bg-slate-950/18',
                disabled && 'cursor-not-allowed opacity-45',
              )}
            >
              <p className="flex items-center gap-2 text-[14px] font-black leading-4 text-slate-100">
                <Icon className={cn('h-5 w-5', tone === 'green' && 'text-emerald-300', tone === 'blue' && 'text-sky-300', tone === 'cyan' && 'text-cyan-300')} />
                {target.label}
              </p>
              <p className="mt-1 text-[11px] leading-4 text-slate-500">{target.description}</p>
            </button>
            )
          })}
        </div>
      </ConfigBlock>

      <ConfigBlock title="参数配置" helper="自定义生效">
        <div className="workspace-parameter-sliders grid gap-5">
          <SliderRow label="扰动强度（ε）" value={formatParameterValue(epsilon, epsilonRange)} pct={pctFromRange(epsilon, epsilonRange)} min={epsilonRange.min} max={epsilonRange.max} step={epsilonRange.step} numericValue={epsilon} onChange={updateNumber(setEpsilon)} />
          <SliderRow label="优化轮数（Steps）" value={String(steps)} pct={pctFromRange(steps, stepsRange)} min={stepsRange.min} max={stepsRange.max} step={stepsRange.step} numericValue={steps} onChange={updateNumber((value) => setSteps(Math.round(value)))} />
        </div>
        <div className="workspace-model-selects mt-5 grid grid-cols-2 gap-2.5 max-sm:grid-cols-1">
          <ModelSelectSummary label="语义编码器" values={selectedSemanticEncoders} options={semanticOptionItems} onOpen={openModelModal} />
          <ModelSelectSummary label="身份编码器" values={selectedFeatureModels} options={featureOptionItems} onOpen={openModelModal} />
        </div>
        <div className="workspace-advanced-row mt-6 border-t border-cyan-300/10 pt-3">
          <button type="button" onClick={() => setLambdaModalOpen(true)} className="flex min-w-0 flex-1 items-center justify-between text-left text-sm font-bold text-slate-300">
            <span className="inline-flex items-center gap-1">高级选项（<MathText formula="\lambda" className="text-cyan-100" />）</span>
            <ChevronDown className="h-4 w-4 text-cyan-300" />
          </button>
        </div>
        <div className="workspace-parameter-help workspace-hover-card mt-5 rounded-[7px] border border-cyan-300/16 bg-sky-400/10 p-3 text-[12px] leading-5 text-slate-300">
          <p className="font-bold text-cyan-200">参数说明</p>
          <p>
            <MathTerm>ε</MathTerm> 控制保护扰动的最大幅度，Steps 控制优化迭代次数；<MathText formula="\lambda" className="mx-0.5 text-cyan-100" /> 权重在高级选项弹窗中调节。建议先保持默认权重，再根据语义扰动、身份相似度与听感质量进行微调。<br/>
            可以参考保护结果页的调参建议结合实际听感以及克隆效果进行定向调整。
          </p>
        </div>
      </ConfigBlock>
      </div>

      {modelModalOpen ? createPortal((
        <ModelConfigModal
          semanticValues={selectedSemanticEncoders}
          semanticOptions={semanticOptionItems}
          featureValues={selectedFeatureModels}
          featureOptions={featureOptionItems}
          modelTypes={runtimeConfig.modelTypes}
          onToggleSemantic={(value) => toggleOption(value, selectedSemanticEncoders, setSelectedSemanticEncoders)}
          onToggleFeature={(value) => toggleOption(value, selectedFeatureModels, setSelectedFeatureModels)}
          onSelectAllSemantic={() => {
            switchToCustom()
            setSelectedSemanticEncoders(semanticOptions)
          }}
          onSelectAllFeature={() => {
            switchToCustom()
            setSelectedFeatureModels(featureOptions)
          }}
          onClose={() => setModelModalOpen(false)}
        />
      ), document.body) : null}

      {lambdaModalOpen ? createPortal((
        <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="lambda 高级参数">
          <div className="ui-card w-full max-w-[620px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="inline-flex items-center gap-1 text-[20px] font-black text-white">
                  高级选项（<MathText formula="\lambda" className="text-cyan-100" />）
                </h3>
                <p className="mt-1 text-xs text-slate-500">配置联合优化目标中的四个 <MathText formula="\lambda" className="mx-0.5 text-cyan-100" /> 权重项</p>
              </div>
              <button type="button" onClick={() => setLambdaModalOpen(false)} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭 lambda 参数">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mb-4 rounded-[7px] border border-cyan-300/14 bg-cyan-400/[0.055] px-4 py-3 text-center">
              <OptimizationFormula className="text-[18px]" />
            </div>
            <div className="space-y-4">
              <SliderRow label={<LambdaLabel name="sem" text="语义权重" />} labelText="lambda sem 语义权重" value={formatParameterValue(lambdaSem, lambdaSemRange)} pct={pctFromRange(lambdaSem, lambdaSemRange)} min={lambdaSemRange.min} max={lambdaSemRange.max} step={lambdaSemRange.step} numericValue={lambdaSem} onChange={updateNumber(setLambdaSem)} />
              <SliderRow label={<LambdaLabel name="id" text="声音身份权重" />} labelText="lambda id 声音身份权重" value={formatParameterValue(lambdaFeat, lambdaFeatRange)} pct={pctFromRange(lambdaFeat, lambdaFeatRange)} min={lambdaFeatRange.min} max={lambdaFeatRange.max} step={lambdaFeatRange.step} numericValue={lambdaFeat} onChange={updateNumber(setLambdaFeat)} />
              <SliderRow label={<LambdaLabel name="psy" text="听感约束" />} labelText="lambda psy 听感约束" value={formatScientificParameterValue(lambdaPsy)} pct={pctFromRange(lambdaPsy, lambdaPsyRange)} min={lambdaPsyRange.min} max={lambdaPsyRange.max} step={lambdaPsyRange.step} numericValue={lambdaPsy} onChange={updateNumber(setLambdaPsy)} />
              <SliderRow label={<LambdaLabel name="2" text="噪声大小约束" />} labelText="lambda 2 噪声大小约束" value={formatParameterValue(lambdaL2, lambdaL2Range)} pct={pctFromRange(lambdaL2, lambdaL2Range)} min={lambdaL2Range.min} max={lambdaL2Range.max} step={lambdaL2Range.step} numericValue={lambdaL2} onChange={updateNumber(setLambdaL2)} />
            </div>
            <div className="mt-5 flex justify-end">
              <button type="button" onClick={() => setLambdaModalOpen(false)} className="cyan-button h-9 min-w-[112px] rounded-[6px] text-sm font-black">
                完成
              </button>
            </div>
          </div>
        </div>
      ), document.body) : null}

      <div className="workspace-task-execution shrink-0 border-t border-cyan-300/10 pt-3">
        <div className="grid gap-3">
          <button
            disabled={running && !configurationChangedSinceSubmission}
            onClick={() => {
              setConfigurationChangedSinceSubmission(false)
              onStart(buildPayload())
            }}
            className="workspace-hover-control cyan-button inline-flex h-[50px] items-center justify-center gap-2 rounded-[7px] text-[16px] font-black disabled:opacity-60"
          >
            {running && !configurationChangedSinceSubmission ? <Loader2 className="h-5 w-5 animate-spin" /> : <ShieldCheck className="h-5 w-5" />}
            开始生成保护音频
          </button>
        </div>
        <p className="mt-3 text-center text-xs text-slate-500">
          {measuredAverageStepSec === null
            ? '预计耗时：完成一次保护任务后可按真实单步耗时估算'
            : `预计耗时：单步 ${measuredAverageStepSec.toFixed(2)} s，总时长预计 ${(steps * measuredAverageStepSec).toFixed(2)} s`}
        </p>
      </div>
    </section>
  )
}

function ConfigBlock({ title, helper, children }: { title: string; helper?: string; children: ReactNode }) {
  return (
    <div className="workspace-config-block mb-8">
      <h3 className="mb-3 flex items-center gap-2 text-[15px] font-black text-slate-200">
        {title}
        {helper ? <span className="text-xs font-normal text-slate-500">{helper}</span> : null}
      </h3>
      {children}
    </div>
  )
}

function SliderRow({
  label,
  labelText,
  value,
  pct,
  min,
  max,
  step,
  numericValue,
  onChange,
  compact,
}: {
  label: ReactNode
  labelText?: string
  value: string
  pct: number
  min: number
  max: number
  step: number
  numericValue: number
  onChange: (value: number) => void
  compact?: boolean
}) {
  const parameterName = labelText ?? (typeof label === 'string' ? label : '参数')
  const [manualInputOpen, setManualInputOpen] = useState(false)
  const [manualInputValue, setManualInputValue] = useState('')
  const [manualInputError, setManualInputError] = useState<string | null>(null)
  const manualInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!manualInputOpen) return
    const timeoutId = window.setTimeout(() => {
      manualInputRef.current?.focus()
      manualInputRef.current?.select()
    }, 0)
    return () => window.clearTimeout(timeoutId)
  }, [manualInputOpen])

  const openManualInput = () => {
    setManualInputValue(value)
    setManualInputError(null)
    setManualInputOpen(true)
  }

  const closeManualInput = () => {
    setManualInputOpen(false)
    setManualInputError(null)
  }

  const confirmManualInput = () => {
    const normalized = manualInputValue.trim().replace(',', '.')
    if (!normalized) {
      setManualInputError('请输入参数值。')
      return
    }
    const next = Number(normalized)
    if (!Number.isFinite(next)) {
      setManualInputError('请输入有效的数字，例如 0.03。')
      return
    }
    if (next < min || next > max) {
      setManualInputError(`请输入 ${min} 至 ${max} 范围内的数值。`)
      return
    }
    if (!isAlignedToStep(next, min, step)) {
      setManualInputError(`请输入符合步长 ${step} 的数值。`)
      return
    }
    onChange(next)
    closeManualInput()
  }

  return (
    <>
      <div className={cn('parameter-slider-row workspace-hover-row grid items-center text-sm', compact ? 'grid-cols-[minmax(96px,112px)_minmax(52px,1fr)_88px] gap-2' : 'grid-cols-[minmax(124px,164px)_minmax(68px,1fr)_112px] gap-3')}>
        <span className="whitespace-nowrap text-slate-300">{label}</span>
        <div className="relative h-5">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={numericValue}
            onChange={(event) => onChange(Number(event.target.value))}
            className="absolute inset-0 z-10 h-5 w-full cursor-pointer opacity-0"
            aria-label={labelText ?? (typeof label === 'string' ? label : '参数滑块')}
          />
          <div className="config-slider-track absolute left-0 right-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-slate-700">
          <div className="config-slider-fill relative h-full rounded-full bg-cyan-400" style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}>
            <span className="config-slider-thumb absolute right-0 top-1/2 h-4 w-4 -translate-y-1/2 translate-x-1/2 rounded-full bg-cyan-300 shadow-[0_0_14px_rgba(34,211,238,0.8)]" />
          </div>
        </div>
        </div>
        <div className="parameter-number-field relative h-9 min-w-0">
          <button
            type="button"
            onClick={openManualInput}
            className="parameter-number-input h-full w-full min-w-0 cursor-text rounded-[6px] border border-cyan-300/16 bg-slate-950/32 py-1.5 font-mono tabular-nums text-slate-200 outline-none transition hover:border-cyan-300/40 focus:border-cyan-300"
            aria-label={`${parameterName} 手动输入`}
            aria-haspopup="dialog"
            aria-expanded={manualInputOpen}
          >
            {value}
          </button>
          <div className="parameter-number-stepper absolute inset-y-px right-px z-10 grid w-7 grid-rows-2 overflow-hidden rounded-r-[5px] border-l border-cyan-300/12 bg-slate-950/18">
            <button type="button" onClick={() => onChange(clampToRange(numericValue + step, { min, max, step }))} className="grid place-items-center text-slate-400 hover:bg-cyan-300/10 hover:text-cyan-200" aria-label={`${parameterName} 增加`}>
              <ChevronUp className="h-3 w-3" />
            </button>
            <button type="button" onClick={() => onChange(clampToRange(numericValue - step, { min, max, step }))} className="grid place-items-center border-t border-cyan-300/10 text-slate-400 hover:bg-cyan-300/10 hover:text-cyan-200" aria-label={`${parameterName} 减少`}>
              <ChevronDown className="h-3 w-3" />
            </button>
          </div>
        </div>
      </div>

      {manualInputOpen ? createPortal(
        <div
          className="fixed inset-0 z-[120] grid place-items-center bg-slate-950/78 px-4 backdrop-blur-sm"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closeManualInput()
          }}
        >
          <form
            className="ui-card w-full max-w-[430px] border border-cyan-300/20 bg-slate-950 p-5 shadow-[0_28px_90px_rgba(0,0,0,0.62)]"
            role="dialog"
            aria-modal="true"
            aria-label={`${parameterName} 数值输入`}
            onSubmit={(event) => {
              event.preventDefault()
              confirmManualInput()
            }}
            onKeyDown={(event) => {
              if (event.key === 'Escape') closeManualInput()
            }}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-[18px] font-black text-white">输入{parameterName}</h3>
                <p className="mt-1 text-xs text-slate-400">完整输入数值后再确认，输入过程中不会修改当前参数。</p>
              </div>
              <button type="button" onClick={closeManualInput} className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-cyan-300/16 text-slate-300 hover:border-cyan-300/35 hover:text-white" aria-label="关闭参数输入">
                <X className="h-4 w-4" />
              </button>
            </div>
            <label className="mt-5 block text-sm font-bold text-slate-200">
              参数值
              <input
                ref={manualInputRef}
                type="text"
                inputMode={step < 1 ? 'decimal' : 'numeric'}
                value={manualInputValue}
                onChange={(event) => {
                  setManualInputValue(event.target.value)
                  if (manualInputError) setManualInputError(null)
                }}
                aria-invalid={Boolean(manualInputError)}
                className="mt-2 h-11 w-full rounded-[7px] border border-cyan-300/20 bg-slate-900 px-3 font-mono text-base tabular-nums text-white outline-none focus:border-cyan-300"
                placeholder="例如 0.03"
              />
            </label>
            <div className="mt-2 min-h-5 text-xs" aria-live="polite">
              {manualInputError
                ? <span className="text-rose-300">{manualInputError}</span>
                : <span className="text-slate-500">允许范围：{min} 至 {max}</span>}
            </div>
            <div className="mt-5 grid grid-cols-2 gap-3">
              <button type="button" onClick={closeManualInput} className="h-10 rounded-[7px] border border-slate-600 bg-slate-900 text-sm font-black text-slate-200 hover:border-slate-400">
                取消
              </button>
              <button type="submit" className="cyan-button h-10 rounded-[7px] text-sm font-black">
                确认
              </button>
            </div>
          </form>
        </div>,
        document.body,
      ) : null}
    </>
  )
}

function MathTerm({ children }: { children: ReactNode }) {
  return <span className="font-serif italic tracking-normal text-cyan-100">{children}</span>
}

function LambdaLabel({ name, text }: { name: string; text: string }) {
  const formula = name === '2' ? '\\lambda_2' : `\\lambda_{\\mathrm{${name}}}`
  return (
    <span className="inline-flex items-center">
      <MathText formula={formula} className="text-cyan-100" />
      <span className="ml-1">（{text}）</span>
    </span>
  )
}

function OptimizationFormula({ className }: { className?: string }) {
  return (
    <MathText
      formula="L = \lambda_{\mathrm{id}} L_{\mathrm{id}} + \lambda_{\mathrm{sem}} L_{\mathrm{sem}} + \lambda_{\mathrm{psy}} L_{\mathrm{psy}} + \lambda_2 \lVert \delta \rVert_2"
      className={cn('text-slate-100', className)}
    />
  )
}

function selectionSummary(values: string[], options: UiModelOption[]) {
  if (!options.length) return '暂未提供'
  if (values.length >= options.length) return `默认全选 · ${options.length}`
  return `已选 · ${values.length}/${options.length}`
}

function ModelSelectSummary({ label, values, options, onOpen }: { label: string; values: string[]; options: UiModelOption[]; onOpen: () => void }) {
  return (
    <div className="workspace-model-select-summary workspace-hover-card min-h-[64px] rounded-[7px] border border-cyan-300/14 bg-slate-950/24 px-3 py-2.5">
      <div className="flex h-full items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate whitespace-nowrap text-sm font-black leading-5 text-slate-200">{label}</p>
          <p className="mt-1 truncate whitespace-nowrap text-xs leading-4 text-slate-500">{selectionSummary(values, options)}</p>
        </div>
        <button
          type="button"
          onClick={onOpen}
          className="workspace-hover-control grid h-9 w-9 shrink-0 place-items-center rounded-[6px] border border-cyan-300/18 bg-cyan-400/10 text-cyan-100 hover:bg-cyan-400/16 disabled:cursor-not-allowed disabled:opacity-45"
          disabled={!options.length}
          aria-label={`配置${label}`}
          title={`配置${label}`}
        >
          <SlidersHorizontal className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}

function ModelConfigModal({
  semanticValues,
  semanticOptions,
  featureValues,
  featureOptions,
  modelTypes,
  onToggleSemantic,
  onToggleFeature,
  onSelectAllSemantic,
  onSelectAllFeature,
  onClose,
}: {
  semanticValues: string[]
  semanticOptions: UiModelOption[]
  featureValues: string[]
  featureOptions: UiModelOption[]
  modelTypes?: CapabilitiesResponse['modelTypes']
  onToggleSemantic: (value: string) => void
  onToggleFeature: (value: string) => void
  onSelectAllSemantic: () => void
  onSelectAllFeature: () => void
  onClose: () => void
}) {
  const [informationModel, setInformationModel] = useState<UiModelOption | null>(null)
  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="防护模型配置">
      <div className="ui-card w-full max-w-[680px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-[20px] font-black text-white">模型配置</h3>
            <p className="mt-1 text-xs text-slate-500">选择参与防护的模型，可同时选择多个</p>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭模型配置">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <MultiChoiceGroup title="语义编码器" values={semanticValues} options={semanticOptions} onToggle={onToggleSemantic} onSelectAll={onSelectAllSemantic} onInformation={setInformationModel} />
          <MultiChoiceGroup title="身份编码器" values={featureValues} options={featureOptions} onToggle={onToggleFeature} onSelectAll={onSelectAllFeature} onInformation={setInformationModel} />
        </div>
        <div className="mt-5 flex justify-end">
          <button type="button" onClick={onClose} className="cyan-button h-10 min-w-[116px] rounded-[7px] text-sm font-black">
            完成
          </button>
        </div>
      </div>
      <ModelInformationModal model={informationModel} modelTypes={modelTypes} onClose={() => setInformationModel(null)} />
    </div>
  )
}

function MultiChoiceGroup({ title, values, options, onToggle, onSelectAll, onInformation }: { title: string; values: string[]; options: UiModelOption[]; onToggle: (value: string) => void; onSelectAll: () => void; onInformation: (option: UiModelOption) => void }) {
  return (
    <div className="rounded-[8px] border border-cyan-300/14 bg-slate-950/24 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-black text-slate-100">{title}</p>
          <p className="mt-1 text-xs text-slate-500">{selectionSummary(values, options)}</p>
        </div>
        <button
          type="button"
          onClick={onSelectAll}
          className="h-8 rounded-[6px] border border-cyan-300/16 bg-white/[0.035] px-3 text-xs font-bold text-slate-300 hover:text-white"
        >
          全选
        </button>
      </div>
      <div className="max-h-[260px] space-y-2 overflow-y-auto pr-1">
        {options.map((option) => {
          const selected = values.includes(option.value)
          const unavailable = option.status !== undefined && option.status !== 'available'
          return (
            <div
              key={option.value}
              className={cn(
                'flex h-10 w-full items-center justify-between gap-3 rounded-[6px] border px-3 text-left text-sm font-bold transition',
                selected ? 'border-cyan-300 bg-cyan-400/12 text-cyan-100' : 'border-cyan-300/14 bg-slate-950/45 text-slate-400',
                unavailable && 'opacity-60',
              )}
            >
              <button type="button" disabled={unavailable} title={unavailable ? '当前模型暂不可用' : option.name ?? option.label ?? option.value} onClick={() => onToggle(option.value)} className={cn('flex min-w-0 flex-1 items-center justify-between gap-3', unavailable && 'cursor-not-allowed')}>
                <span className="min-w-0 truncate">{option.label}</span>
                <span className={cn('grid h-4 w-4 shrink-0 place-items-center rounded border', selected ? 'border-cyan-300 bg-cyan-300 text-slate-950' : 'border-slate-600')}>
                  {selected ? <CheckSquare className="h-3 w-3" /> : null}
                </span>
              </button>
              {option.information ? (
                <button type="button" onClick={(event) => { event.preventDefault(); event.stopPropagation(); onInformation(option) }} className="grid h-7 w-7 shrink-0 place-items-center rounded-[5px] border border-cyan-300/14 text-cyan-200 hover:bg-cyan-300/10" aria-label={`查看 ${option.label} 详情`} title="查看模型详情">
                  <Search className="h-3.5 w-3.5" />
                </button>
              ) : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function TinyWave({ color, className }: { color: string; className?: string }) {
  return (
    <svg viewBox="0 0 400 100" className={cn('h-full w-full', className)} preserveAspectRatio="none">
      <line x1="0" x2="400" y1="50" y2="50" stroke={color} strokeOpacity="0.22" />
      {Array.from({ length: 82 }, (_, index) => {
        const height = 8 + Math.abs(Math.sin(index * 0.65) * 38) + (index % 5) * 4
        return <rect key={index} x={index * 4.85} y={(100 - height) / 2} width="2.2" height={height} rx="1" fill={color} opacity={0.35 + (index % 4) * 0.12} />
      })}
    </svg>
  )
}
