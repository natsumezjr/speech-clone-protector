import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Box,
  CheckSquare,
  ChevronDown,
  Copy,
  FileAudio,
  FileText,
  Gauge,
  Headphones,
  Info,
  Loader2,
  Mic,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  StopCircle,
  UploadCloud,
  Waves,
  X,
} from 'lucide-react'

import { createProtectionTask, uploadFile } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { ProtectionTaskRequest, TaskStatusResponse } from '@/types/task'
import { apiBaseUrl, isBackendMode, isMockMode } from '@/config/runtime'
import { cn } from '@/lib/utils'
import { AudioPlayer } from '@/components/audio/AudioPlayer'
import { formatDurationSeconds, getAudioDuration, getRecordedExtension, getRecorderMimeType, readAudioDuration } from '@/utils/audio'
import { formatBytes, shortHash } from '@/utils/format'

const MAX_AUDIO_SIZE = 200 * 1024 * 1024

const defaultPayload: ProtectionTaskRequest = {
  mode: 'custom',
  targets: ['semantic', 'timbre'],
  semantic: { enabled: true, asrModel: 'Paraformer-large', encoders: ['S3', 'Whisper', 'MFCC'], lambdaSemantic: 0.15 },
  timbre: { enabled: true, mode: 'untargeted', encoders: ['HuBERT'], lambdaTimbre: 0.2 },
  psychoacoustic: { enabled: true, lambdaPsy: 0.15 },
  optimization: { epsilon: 0.08, steps: 20 },
}

function ProtectionLoadingModal({ progress }: { progress: number }) {
  return (
    <div className="fixed inset-0 z-[80] grid place-items-center bg-slate-950/60 px-4 backdrop-blur-sm" role="alertdialog" aria-live="assertive" aria-label="正在生成保护音频">
      <div className="ui-card w-full max-w-sm p-8 text-center">
        <div className="mx-auto grid h-16 w-16 place-items-center rounded-full border border-cyan-300/18 bg-cyan-400/10">
          <Loader2 className="h-9 w-9 animate-spin text-cyan-300" />
        </div>
        <h3 className="mt-5 text-lg font-black text-white">正在生成保护音频</h3>
        <p className="mt-2 text-sm leading-6 text-slate-300">系统正在执行语义防护、Feature 特征防护与听感约束，请稍候。</p>
        <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-800">
          <div className="h-full rounded-full bg-cyan-400 transition-all duration-300" style={{ width: `${Math.max(8, Math.round(progress * 100))}%` }} />
        </div>
        <p className="mt-2 font-mono text-xs text-slate-500">{Math.round(progress * 100)}%</p>
      </div>
    </div>
  )
}

function revokeObjectUrl(url?: string) {
  if (url?.startsWith('blob:')) URL.revokeObjectURL(url)
}

export function WorkspacePage() {
  const navigate = useNavigate()
  const [progress, setProgress] = useState(0)
  const [running, setRunning] = useState(false)
  const [taskId, setTaskId] = useState<string>()
  const [taskStatus, setTaskStatus] = useState<TaskStatusResponse | null>(null)

  const pushToast = useAppStore((state) => state.pushToast)
  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const setCurrentTaskStatus = useTaskStore((state) => state.setCurrentTaskStatus)

  const startTask = async (payload: ProtectionTaskRequest = defaultPayload) => {
    if (isBackendMode && !uploadedFile?.fileId) {
      pushToast({ kind: 'error', title: '无法创建任务', description: 'Backend 模式下请先上传音频并等待后端返回 fileId。' })
      return
    }

    try {
      setRunning(true)
      setProgress(0)
      setTaskStatus(null)

      const created = await createProtectionTask({ ...payload, fileId: uploadedFile?.fileId ?? payload.fileId })
      setTaskId(created.taskId)
      pushToast({ kind: 'success', title: '任务已创建', description: isMockMode ? 'Mock 演示结果已准备。' : `任务 ID：${created.taskId}` })

      if (isMockMode) {
        const now = new Date().toLocaleString('zh-CN', { hour12: false })
        const status: TaskStatusResponse = {
          taskId: created.taskId,
          status: 'completed',
          progress: 1,
          stage: 'report_generation',
          message: 'Mock 演示任务完成',
          createdAt: now,
          updatedAt: now,
          error: null,
        }
        setProgress(1)
        setRunning(false)
        setCurrentTaskStatus(status)
        setTaskStatus(status)
        navigate(`/results/${created.taskId}`)
        return
      }

      setRunning(false)
      navigate(`/results/${created.taskId}`)
    } catch (error) {
      setRunning(false)
      pushToast({ kind: 'error', title: '任务创建失败', description: error instanceof Error ? error.message : '请检查后端服务。' })
    }
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-[448px_1fr_514px] gap-3 max-2xl:grid-cols-[448px_1fr_514px] max-xl:grid-cols-1">
        <AudioAccessCard />
        <StrategyConfigCard running={running} onStart={(payload) => void startTask(payload)} onMock={() => navigate('/results/mock-task-001')} />
        <ArchitectureCard />
      </div>

      <TaskStatusStrip progress={progress} running={running} taskId={taskId} status={taskStatus} />
      {running ? <ProtectionLoadingModal progress={progress} /> : null}
    </div>
  )
}

function AudioAccessCard() {
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
    if (file.size > MAX_AUDIO_SIZE) {
      pushToast({ kind: 'error', title: '文件过大', description: '单文件大小不能超过 200MB。' })
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

      if (isMockMode) {
        pushToast({ kind: 'success', title: '音频已接入', description: `${file.name} 可立即播放预览。` })
        return
      }

      const backendMeta = await uploadFile(file)
      setUploadedFile({ ...localMeta, ...backendMeta, objectUrl, rawFile: file })
      pushToast({ kind: 'success', title: '音频已上传', description: backendMeta.fileId ? `fileId：${backendMeta.fileId}` : `${file.name} 已进入防护队列。` })
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
        const name = `recorded_voice_${Date.now()}.${extension}`
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

  return (
    <section className="ui-card min-h-[730px] p-4">
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
          className="mt-5 grid min-h-[227px] place-items-center rounded-[12px] border border-dashed border-cyan-300/55 bg-sky-400/5 text-center transition hover:border-cyan-200 hover:bg-cyan-400/10"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault()
            void handleFile(event.dataTransfer.files[0])
          }}
        >
          <div className="flex w-full flex-col items-center px-6 pt-6 pb-6 text-center">
            <UploadCloud className="mb-4 h-14 w-14 text-cyan-300" />
            <p className="text-[18px] font-black text-white">拖拽音频文件到此处，或点击上传</p>
            <p className="mt-3 text-[15px] leading-6 text-slate-300">
              支持 .wav / .flac / .mp3 / .m4a / .webm
              <br />
              单文件 ≤ 200MB
            </p>
            <input ref={inputRef} type="file" accept=".wav,.mp3,.flac,.m4a,.webm,audio/*" className="hidden" onChange={(event) => void handleFile(event.target.files?.[0])} />
            <span onClick={() => inputRef.current?.click()} className="mt-6 inline-flex h-11 min-w-[180px] items-center justify-center gap-2 rounded-full bg-cyan-300 px-5 text-[17px] font-medium text-slate-950 shadow-[0_0_28px_rgba(34,211,238,0.25)] cursor-pointer">
              <UploadCloud className="h-5 w-5" />
              选择文件
            </span>
          </div>
        </div>
      ) : (
        <div className="mt-5 grid min-h-[227px] place-items-center rounded-[12px] border border-dashed border-cyan-300/55 bg-sky-400/5 text-center transition hover:border-cyan-200 hover:bg-cyan-400/10">
          <div className="flex w-full flex-col items-center px-6 pt-6 pb-6 text-center">
            <button
              type="button"
              onClick={toggleRecording}
              className={cn(
                'grid h-[4.5rem] w-[4.5rem] place-items-center rounded-full border-2 transition',
                recording ? 'border-red-300 bg-red-400/18 text-red-200 shadow-[0_0_24px_rgba(248,113,113,0.35)]' : 'border-cyan-300/40 bg-cyan-400/12 text-cyan-200',
              )}
              aria-pressed={recording}
            >
              {recording ? <StopCircle className="h-8 w-8" /> : <Mic className="h-8 w-8" />}
            </button>
            <p className="mt-4 text-[18px] font-black text-white">{recording ? '点击停止录音' : '点击开始录音'}</p>
            <p className="mt-2 font-mono text-[20px] font-black text-cyan-100">{formatDurationSeconds(recordingSec)}</p>
            <p className="mt-3 text-[15px] text-slate-300">录音结束后会自动作为当前音频输入，并在 Backend 模式上传。</p>
            <TinyWave color={recording ? '#f87171' : '#00aef0'} className="mx-auto mt-5 h-9 w-full max-w-[330px]" />
          </div>
        </div>
      )}

      <div className="mt-9 border-t border-cyan-300/10 pt-3">
        <h3 className="text-[15px] font-bold text-slate-300">已上传文件</h3>
      </div>
      <div className="mt-2 max-h-[420px] overflow-y-auto rounded-[8px] border border-cyan-300/12 bg-[#07192d]/85 p-4">
        <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
            <div>
              <p className="font-bold text-white">{displayFile.filename}</p>
            </div>
          </div>
          <span className={cn('rounded-full px-3 py-1 text-xs font-bold', uploadedFile ? 'bg-emerald-400/12 text-emerald-300' : 'bg-slate-500/12 text-slate-400')}>
            {uploadedFile ? '就绪' : '待上传'}
          </span>
        </div>
        <AudioPlayer
          src={uploadedFile?.objectUrl ?? uploadedFile?.audioUrl}
          title="播放预览"
          filename={uploadedFile?.filename}
          disabledReason="请选择或拖拽音频文件后预览"
        />
        <TinyWave color="#00aef0" className="h-[82px]" />
        <div className="mt-1 flex justify-between font-mono text-xs text-slate-400">
          <span>00:00</span>
          <span>{formatDurationSeconds(duration)}</span>
        </div>
        <div className="mt-5 grid grid-cols-3 gap-y-6 border-t border-cyan-300/10 pt-4 text-sm">
          {[
            ['文件名', displayFile.filename],
            ['时长', duration ? `${duration.toFixed(2)}s` : '待解析'],
            ['采样率', uploadedFile?.sampleRate ? `${uploadedFile.sampleRate.toLocaleString()} Hz` : '待后端解析'],
            ['声道', uploadedFile?.channels ? `${uploadedFile.channels}` : '待后端解析'],
            ['位深', uploadedFile?.bitDepth ? `${uploadedFile.bitDepth} bit` : '待后端解析'],
            ['大小', uploadedFile ? formatBytes(uploadedFile.sizeBytes) : '-'],
            ['格式', displayFile.format],
            ['上传时间', uploadedFile?.uploadedAt ?? '-'],
            ['指纹', uploadedFile?.fingerprint ?? '-'],
          ].map(([label, value]) => (
            <div key={label} className="min-w-0">
              <p className="text-slate-500">{label}</p>
              <p className="mt-1 truncate font-semibold text-slate-200">{value}</p>
            </div>
          ))}
        </div>
      </div>

    </section>
  )
}

function StrategyConfigCard({ running, onStart, onMock }: { running: boolean; onStart: (payload: ProtectionTaskRequest) => void; onMock: () => void }) {
  const pushToast = useAppStore((state) => state.pushToast)
  const [selectedMode, setSelectedMode] = useState(3)
  const [selectedTargets, setSelectedTargets] = useState<Array<'semantic' | 'timbre' | 'joint'>>(['semantic'])
  const [lambdaModalOpen, setLambdaModalOpen] = useState(false)
  const [epsilon, setEpsilon] = useState(0.08)
  const [steps, setSteps] = useState(20)
  const [lambdaSem, setLambdaSem] = useState(0.18)
  const [lambdaFeat, setLambdaFeat] = useState(0.22)
  const [lambdaPsy, setLambdaPsy] = useState(0.12)
  const [lambdaL2, setLambdaL2] = useState(0.01)
  const [asrModel, setAsrModel] = useState('Paraformer-large')
  const [featureModel, setFeatureModel] = useState('HuBERT (Base)')

  const toggleTarget = (target: 'semantic' | 'timbre' | 'joint') => {
    setSelectedTargets((current) => {
      if (current.includes(target)) return current.length === 1 ? current : current.filter((item) => item !== target)
      return [...current, target]
    })
  }

  const buildPayload = (): ProtectionTaskRequest => {
    const semanticEnabled = selectedTargets.includes('semantic') || selectedTargets.includes('joint')
    const featureEnabled = selectedTargets.includes('timbre') || selectedTargets.includes('joint')
    return {
      mode: selectedMode === 0 ? 'standard' : selectedMode === 1 ? 'strong' : selectedMode === 2 ? 'high_fidelity' : 'custom',
      targets: [semanticEnabled ? 'semantic' : null, featureEnabled ? 'timbre' : null].filter(Boolean) as Array<'semantic' | 'timbre'>,
      semantic: { enabled: semanticEnabled, asrModel, encoders: ['S3', 'Whisper', 'MFCC'], lambdaSemantic: lambdaSem },
      timbre: { enabled: featureEnabled, mode: 'untargeted', encoders: [featureModel], lambdaTimbre: lambdaFeat },
      psychoacoustic: { enabled: true, lambdaPsy },
      optimization: { epsilon, steps },
    }
  }

  return (
    <section className="ui-card min-h-[730px] p-5">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-[21px] font-black text-white">
          防护策略配置
          <Info className="h-4 w-4 text-slate-500" />
        </h2>
        <button
          type="button"
          onClick={() => pushToast({ kind: 'info', title: '策略模板', description: '当前页面已内置标准、强保护、高保真与高级自定义模板。' })}
          className="rounded-[6px] border border-cyan-300/14 bg-white/[0.03] px-3 py-2 text-sm text-slate-300"
        >
          <Settings className="mr-1 inline h-4 w-4" />
          策略模板
        </button>
      </div>

      <ConfigBlock title="保护模式">
        <div className="grid grid-cols-4 gap-2.5">
          {[
            ['标准保护', '平衡安全与听感'],
            ['强保护', '更强安全性，略降听感'],
            ['高保真', '更优听感，安全性适中'],
            ['高级自定义', '自由调整参数'],
          ].map(([name, sub], index) => (
            <button
              key={name}
              type="button"
              onClick={() => setSelectedMode(index)}
              className={cn(
                'h-[70px] rounded-[6px] border bg-slate-950/18 px-3 text-left',
                index === selectedMode ? 'border-cyan-400 bg-cyan-400/8 text-cyan-200' : 'border-cyan-300/12 text-slate-300',
              )}
            >
              <p className="font-black">{name}</p>
              <p className="mt-1 text-xs text-slate-500">{sub}</p>
            </button>
          ))}
        </div>
      </ConfigBlock>

      <ConfigBlock title="防护目标（可多选）">
        <div className="grid grid-cols-3 gap-2.5">
          {[
            ['语义防护', '降低 ASR/LLM 理解概率', ShieldCheck, 'green', 'semantic'],
            ['Feature 特征防护', '阻断声学特征重建', Waves, 'blue', 'timbre'],
            ['联合防护（推荐）', '语义 + 特征联合防护', ShieldCheck, 'cyan', 'joint'],
          ].map(([name, sub, Icon, tone, target]) => {
            const selected = selectedTargets.includes(target as 'semantic' | 'timbre' | 'joint')
            return (
            <button
              key={String(name)}
              type="button"
              onClick={() => toggleTarget(target as 'semantic' | 'timbre' | 'joint')}
              className={cn(
                'h-[74px] rounded-[6px] border px-3 py-2 text-left',
                selected ? 'border-cyan-400 bg-cyan-400/10' : 'border-cyan-300/12 bg-slate-950/18',
              )}
            >
              <p className="flex items-center gap-2 text-[14px] font-black leading-4 text-slate-100">
                <Icon className={cn('h-5 w-5', tone === 'green' && 'text-emerald-300', tone === 'blue' && 'text-sky-300', tone === 'cyan' && 'text-cyan-300')} />
                {name as string}
              </p>
              <p className="mt-1 text-[11px] leading-4 text-slate-500">{sub as string}</p>
            </button>
            )
          })}
        </div>
      </ConfigBlock>

      <ConfigBlock title="参数配置" helper="对抗优化与听感约束相关参数（高级自定义模式生效）">
        <div className="space-y-5">
          <SliderRow label="ε（扰动强度）" value={epsilon.toFixed(2)} pct={(epsilon / 0.12) * 100} min={0.01} max={0.12} step={0.01} numericValue={epsilon} onChange={setEpsilon} />
          <SliderRow label="优化轮数（Steps）" value={String(steps)} pct={(steps / 40) * 100} min={8} max={40} step={1} numericValue={steps} onChange={setSteps} />
        </div>
        <div className="mt-5 grid grid-cols-2 gap-3">
          <SelectInput label="ASR 模型" value={asrModel} onChange={setAsrModel} options={['Paraformer-large', 'Whisper-large-v3', 'Conformer-CTC']} />
          <SelectInput label="Feature 编码器（声学特征分支）" value={featureModel} onChange={setFeatureModel} options={['HuBERT (Base)', 'WavLM-large', 'ECAPA-TDNN']} />
        </div>
        <button
          type="button"
          onClick={() => setLambdaModalOpen(true)}
          className="mt-6 flex w-full items-center justify-between border-t border-cyan-300/10 pt-3 text-left text-sm font-bold text-slate-300"
        >
          <span>高级选项（lamda）</span>
          <ChevronDown className="h-4 w-4 text-cyan-300" />
        </button>
        <div className="mt-5 rounded-[7px] border border-cyan-300/16 bg-sky-400/10 p-3 text-[12px] leading-5 text-slate-300">
          <p className="font-bold text-cyan-200">参数说明</p>
          <p>
            <MathTerm>ε</MathTerm> 控制保护扰动的最大幅度，Steps 控制优化迭代次数；lamda 权重在高级选项弹窗中调节。建议先保持默认权重，再根据 ASR 干扰、Feature 相似度与听感质量进行微调。
          </p>
        </div>
      </ConfigBlock>

      {lambdaModalOpen ? (
        <div className="fixed inset-0 z-[90] grid place-items-center bg-slate-950/68 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="lamda 高级参数">
          <div className="ui-card w-full max-w-[620px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.46)]">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="text-[20px] font-black text-white">高级选项（lamda）</h3>
                <p className="mt-1 text-xs text-slate-500">配置联合优化目标中的四个 λ 权重项</p>
              </div>
              <button type="button" onClick={() => setLambdaModalOpen(false)} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭 lamda 参数">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mb-4 rounded-[7px] border border-cyan-300/14 bg-cyan-400/[0.055] px-4 py-3 text-center">
              <OptimizationFormula className="text-[18px]" />
            </div>
            <div className="space-y-4">
              <SliderRow label={<LambdaLabel name="sem" text="语义权重" />} labelText="lambda sem 语义权重" value={lambdaSem.toFixed(2)} pct={(lambdaSem / 0.3) * 100} min={0.01} max={0.3} step={0.01} numericValue={lambdaSem} onChange={setLambdaSem} />
              <SliderRow label={<LambdaLabel name="feat" text="Feature 权重" />} labelText="lambda feat Feature 权重" value={lambdaFeat.toFixed(2)} pct={(lambdaFeat / 0.35) * 100} min={0.01} max={0.35} step={0.01} numericValue={lambdaFeat} onChange={setLambdaFeat} />
              <SliderRow label={<LambdaLabel name="psy" text="听感约束" />} labelText="lambda psy 听感约束" value={lambdaPsy.toFixed(2)} pct={(lambdaPsy / 0.25) * 100} min={0.01} max={0.25} step={0.01} numericValue={lambdaPsy} onChange={setLambdaPsy} />
              <SliderRow label={<LambdaLabel name="2" text="L2 正则" />} labelText="lambda 2 L2 正则" value={lambdaL2.toFixed(2)} pct={(lambdaL2 / 0.08) * 100} min={0.0} max={0.08} step={0.01} numericValue={lambdaL2} onChange={setLambdaL2} />
            </div>
            <div className="mt-5 flex justify-end">
              <button type="button" onClick={() => setLambdaModalOpen(false)} className="cyan-button h-9 min-w-[112px] rounded-[6px] text-sm font-black">
                完成
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <div className="mt-5">
        <h3 className="mb-3 text-[15px] font-black text-white">任务执行</h3>
        <div className="grid grid-cols-2 gap-3">
          <button disabled={running} onClick={() => onStart(buildPayload())} className="cyan-button inline-flex h-[50px] items-center justify-center gap-2 rounded-[7px] text-[16px] font-black disabled:opacity-60">
            {running ? <Loader2 className="h-5 w-5 animate-spin" /> : <ShieldCheck className="h-5 w-5" />}
            开始生成保护音频
          </button>
          <button onClick={onMock} className="inline-flex h-[50px] items-center justify-center gap-2 rounded-[7px] border border-cyan-300/18 bg-white/[0.03] text-[16px] font-bold text-slate-200">
            <Box className="h-5 w-5" />
            使用 Mock 数据演示
          </button>
        </div>
        <p className="mt-3 text-center text-xs text-slate-500">预计耗时：30~120 秒（视音频长度与参数复杂度）</p>
      </div>
    </section>
  )
}

function ArchitectureCard() {
  const pushToast = useAppStore((state) => state.pushToast)
  const mode = isMockMode ? 'mock' : 'api'
  const apiStatusLabel = mode === 'api' ? '已配置' : 'Mock'
  const notifyMode = (target: 'mock' | 'api') => {
    if (target === mode) {
      pushToast({ kind: 'info', title: '当前运行模式', description: target === 'api' ? `Backend：${apiBaseUrl}` : 'Mock Client' })
      return
    }
    pushToast({ kind: 'info', title: '需要重启前端', description: `请设置 VITE_DATA_MODE=${target === 'api' ? 'backend' : 'mock'} 后重启 dev server。` })
  }

  return (
    <section className="grid gap-3">
      <div className="ui-card min-h-[550px] p-5">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-[21px] font-black text-white">系统架构概览</h2>
            <p className="mt-2 text-sm text-slate-400">端到端语音克隆防护流程（E2E-VGuard）</p>
          </div>
          <button type="button" onClick={() => pushToast({ kind: 'info', title: '系统架构', description: '左侧上传音频，中间生成保护音频，右侧结果页可执行克隆测试与评估导出。' })} className="text-sm font-bold text-cyan-300">查看详情 ›</button>
        </div>
        <div className="grid grid-cols-[118px_36px_132px_36px_128px] items-center gap-2">
          <ArchBox title="输入音频" sub={<VariableSymbol name="x" />} icon={<Waves className="h-10 w-10 text-sky-200" />} />
          <Arrow />
          <ArchBox title="防护优化引擎" sub="" icon={<ShieldCheck className="h-12 w-12 text-cyan-300" />} active />
          <Arrow />
          <ArchBox title="保护音频" sub={<VariableSymbol name="x" prime />} icon={<Waves className="h-10 w-10 text-sky-200" />} />
        </div>
        <div className="mt-8 grid grid-cols-3 gap-3">
          <Branch title="语义分支" color="green" items={['ASR 系统', '多模型语义编码', '表示空间约束', '...']} />
          <Branch title="Feature 分支" color="blue" items={['Feature Encoder', '声学特征约束', '说话人不可恢复', '...']} />
          <Branch title="听感约束" color="purple" items={['心理声学模型', '掩蔽阈值建模', '听感优化', '...']} />
        </div>
        <div className="mt-6 rounded-[7px] border border-cyan-300/20 bg-sky-400/10 p-4 text-center text-sm text-slate-300">
          <span className="mr-3 text-slate-400">联合优化目标</span>
          <OptimizationFormula />
          <p className="mt-2 text-xs text-slate-400">可以在高级自定义中调整实际 λ 参数</p>
        </div>
      </div>

      <div className="grid grid-cols-[1fr_238px] gap-3">
        <div className="ui-card p-4">
          <h3 className="font-black text-lime-300">当前前端运行模式</h3>
          <p className="mt-2 text-sm leading-6 text-slate-400">运行模式由 VITE_DATA_MODE 控制；当前默认走后端 API，Mock 入口保留用于演示。</p>
          <div className="mt-4 inline-flex rounded-full border border-cyan-300/14 bg-slate-950/30 p-1">
            <button type="button" onClick={() => notifyMode('mock')} className={cn('rounded-full px-5 py-2 text-sm font-black', mode === 'mock' ? 'bg-cyan-400 text-slate-950' : 'text-slate-400')}>Mock 模式</button>
            <button type="button" onClick={() => notifyMode('api')} className={cn('rounded-full px-5 py-2 text-sm font-black', mode === 'api' ? 'bg-cyan-400 text-slate-950' : 'text-slate-400')}>API 模式</button>
          </div>
        </div>
        <div className="ui-card p-4">
          <h3 className="flex items-center justify-between font-black text-white">
            API 状态
            <span className={cn('rounded px-2 py-1 text-xs', mode === 'api' ? 'bg-emerald-400/14 text-emerald-300' : 'bg-cyan-400/14 text-cyan-300')}>{apiStatusLabel}</span>
          </h3>
          {['POST /api/files/upload', 'POST /api/tasks/protect', 'GET /api/tasks/{id}', 'POST /api/tasks/{id}/clone-voice'].map((item) => (
            <p key={item} className="mt-3 flex justify-between border-b border-cyan-300/8 pb-2 text-xs text-slate-300">
              <span>{item.split(' ')[1]}</span>
              <span className={mode === 'api' ? 'text-emerald-300' : 'text-cyan-300'}>{item.split(' ')[0]}</span>
            </p>
          ))}
          <p className="mt-3 flex items-center justify-between text-xs text-slate-400">
            服务地址：{mode === 'api' ? apiBaseUrl : 'Mock Client'}
            <Copy className="h-4 w-4" />
          </p>
        </div>
      </div>
    </section>
  )
}

function TaskStatusStrip({ progress, running, taskId, status }: { progress: number; running: boolean; taskId?: string; status?: TaskStatusResponse | null }) {
  const steps = [
    ['file_preprocess', '文件预处理', FileAudio],
    ['encoder_loading', '编码器加载', Mic],
    ['perturbation_optimization', '扰动优化', SlidersHorizontal],
    ['psychoacoustic_constraint', '心理声学约束', Headphones],
    ['result_evaluation', '结果评估', Gauge],
    ['report_generation', '报告生成', FileText],
  ] as const
  const stageIndex = status?.stage ? steps.findIndex(([stage]) => stage === status.stage) : -1
  const activeIndex = running ? (stageIndex >= 0 ? stageIndex : Math.min(5, Math.floor(progress * steps.length))) : status?.status === 'completed' || status?.status === 'success' ? steps.length - 1 : -1

  return (
    <section className="ui-card grid min-h-[150px] grid-cols-[1fr_492px] gap-4 p-4 max-xl:grid-cols-1">
      <div className="flex flex-col justify-center ml-2.5">
        <div className="flex items-start justify-between gap-2">
          {steps.map(([, label, Icon], index) => (
            <div key={label} className="flex flex-1 items-start">
              <div className="text-center">
                <div className={cn('mx-auto grid h-10 w-10 place-items-center rounded-full border', index <= activeIndex ? 'border-cyan-300 bg-cyan-400/14 text-cyan-200' : 'border-slate-600 text-slate-500')}>
                  <Icon className="h-5 w-5" />
                </div>
                <p className="mt-2 text-sm font-bold text-slate-200">{label}</p>
                <p className="mt-2 text-xs text-slate-500">{index < activeIndex ? '已完成' : index === activeIndex && running ? '处理中' : '等待开始'}</p>
              </div>
              {index < steps.length - 1 ? <div className={cn('mt-5 h-px flex-1', index < activeIndex ? 'bg-cyan-400' : 'bg-cyan-300/18')} /> : null}
            </div>
          ))}
        </div>
        {taskId ? <p className="mt-3 text-xs text-cyan-300">当前任务：{taskId}</p> : null}
        {status?.message ? <p className="mt-1 text-xs text-slate-400">后端状态：{status.message}</p> : null}
      </div>
      <div className="grid grid-cols-[1fr_154px] gap-3">
        <div className="ui-card -ml-[30px] bg-sky-400/7 p-4">
          <h3 className="font-black text-white">结果产物（完成后可下载）</h3>
          <div className="mt-3 grid grid-cols-2 gap-2 text-sm text-slate-300">
            {['保护音频（.wav）', '评估报告（.pdf）', '中间特征（.npz）', '可视化图表（.png）'].map((item) => (
              <label key={item} className="flex items-center gap-2">
                <CheckSquare className="h-4 w-4 text-cyan-300" />
                {item}
              </label>
            ))}
          </div>
        </div>
        <div className="ui-card grid place-items-center p-4 text-center">
          <FileText className="h-10 w-10 text-slate-300" />
          <p className="mt-2 font-black text-white">操作日志</p>
          <p className="text-sm text-slate-400">查看详细日志</p>
        </div>
      </div>
    </section>
  )
}

function ConfigBlock({ title, helper, children }: { title: string; helper?: string; children: ReactNode }) {
  return (
    <div className="mb-8">
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
  return (
    <div className={cn('grid items-center text-sm', compact ? 'grid-cols-[112px_1fr_58px] gap-2' : 'grid-cols-[170px_1fr_74px] gap-4')}>
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
        <div className="absolute left-0 right-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-slate-700">
        <div className="relative h-full rounded-full bg-cyan-400" style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}>
          <span className="absolute right-0 top-1/2 h-4 w-4 -translate-y-1/2 translate-x-1/2 rounded-full bg-cyan-300 shadow-[0_0_14px_rgba(34,211,238,0.8)]" />
        </div>
      </div>
      </div>
      <span className={cn('rounded-[6px] border border-cyan-300/16 bg-slate-950/32 py-1.5 text-center font-mono text-slate-200', compact ? 'px-1.5' : 'px-3')}>{value}</span>
    </div>
  )
}

function MathTerm({ children }: { children: ReactNode }) {
  return <span className="font-serif italic tracking-normal text-cyan-100">{children}</span>
}

function LambdaLabel({ name, text }: { name: string; text: string }) {
  return (
    <span>
      <span className="font-serif italic tracking-normal text-cyan-100">
        λ<sub>{name}</sub>
      </span>
      <span className="ml-1">（{text}）</span>
    </span>
  )
}

function OptimizationFormula({ className }: { className?: string }) {
  return (
    <span className={cn('font-serif italic tracking-normal text-slate-100', className)}>
      <span className="text-cyan-100">L</span>
      <span> = </span>
      <span>λ</span>
      <sub>sem</sub>
      <span>L</span>
      <sub>sem</sub>
      <span> + λ</span>
      <sub>feat</sub>
      <span>L</span>
      <sub>feat</sub>
      <span> + λ</span>
      <sub>psy</sub>
      <span>L</span>
      <sub>psy</sub>
      <span> + λ</span>
      <sub>2</sub>
      <span>∥δ∥</span>
      <sub>2</sub>
    </span>
  )
}

function SelectInput({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <label className="text-sm text-slate-300">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-2 h-9 w-full rounded-[6px] border border-cyan-300/14 bg-slate-950/70 px-3 text-slate-200 outline-none transition focus:border-cyan-300"
      >
        {options.map((option) => (
          <option key={option} value={option} className="bg-slate-950 text-slate-100">
            {option}
          </option>
        ))}
      </select>
    </label>
  )
}

function VariableSymbol({ name, prime }: { name: string; prime?: boolean }) {
  return (
    <span className="font-serif italic tracking-normal text-slate-300">
      {name}
      {prime ? <sup>′</sup> : null}
    </span>
  )
}

function ArchBox({ title, sub, icon, active }: { title: string; sub: ReactNode; icon: ReactNode; active?: boolean }) {
  return (
    <div className={cn('grid h-[118px] place-items-center rounded-[7px] border bg-slate-950/20 text-center', active ? 'border-cyan-400/35 bg-cyan-400/10' : 'border-cyan-300/14')}>
      {icon}
      <p className="font-black text-slate-200">{title}</p>
      {sub ? <p className="text-sm text-slate-400">{sub}</p> : null}
    </div>
  )
}

function Branch({ title, color, items }: { title: string; color: 'green' | 'blue' | 'purple'; items: string[] }) {
  return (
    <div className={cn('rounded-[8px] border p-3 text-center', color === 'green' && 'border-emerald-400/35 bg-emerald-400/10', color === 'blue' && 'border-sky-400/35 bg-sky-400/10', color === 'purple' && 'border-violet-400/35 bg-violet-400/10')}>
      <h3 className={cn('mb-3 text-[18px] font-black', color === 'green' && 'text-emerald-300', color === 'blue' && 'text-sky-300', color === 'purple' && 'text-violet-300')}>{title}</h3>
      <div className="space-y-2">
        {items.map((item) => (
          <div key={item} className="rounded-[6px] border border-white/10 bg-slate-950/18 px-2 py-3 text-sm text-slate-200">
            {item}
          </div>
        ))}
      </div>
    </div>
  )
}

function Arrow() {
  return <div className="text-center text-3xl text-slate-400">→</div>
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
