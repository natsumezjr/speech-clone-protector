import { FileAudio, Mic, PlayCircle, RotateCcw, StopCircle, UploadCloud } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Badge } from '@/components/common/Badge'
import { Button } from '@/components/common/Button'
import { Panel } from '@/components/common/Panel'
import { isMockMode } from '@/config/runtime'
import { uploadFile } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import { formatBytes, shortHash } from '@/utils/format'
import { AudioWaveform } from './AudioWaveform'

const MAX_AUDIO_SIZE = 200 * 1024 * 1024

function formatDuration(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function getRecorderMimeType() {
  if (typeof MediaRecorder === 'undefined') return undefined

  return ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/mpeg'].find((type) => MediaRecorder.isTypeSupported(type))
}

function getRecordedExtension(mimeType: string) {
  if (mimeType.includes('mp4')) return 'm4a'
  if (mimeType.includes('mpeg')) return 'mp3'
  return 'webm'
}

export function AudioUploadPanel() {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const timerRef = useRef<number | null>(null)
  const recordedUrlRef = useRef<string | null>(null)

  const [activeTab, setActiveTab] = useState<'upload' | 'record'>('upload')
  const [recording, setRecording] = useState(false)
  const [recordingSec, setRecordingSec] = useState(0)
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null)
  const [recordedName, setRecordedName] = useState<string | null>(null)

  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const setUploadedFile = useTaskStore((state) => state.setUploadedFile)
  const pushToast = useAppStore((state) => state.pushToast)

  const clearRecordingTimer = () => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  const releaseRecorderStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    recorderRef.current = null
  }

  const clearRecordedPreview = () => {
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current)
      recordedUrlRef.current = null
    }
    setRecordedUrl(null)
    setRecordedName(null)
  }

  useEffect(() => {
    return () => {
      clearRecordingTimer()
      releaseRecorderStream()
      clearRecordedPreview()
    }
  }, [])

  const handleFile = async (file: File) => {
    if (file.size > MAX_AUDIO_SIZE) {
      pushToast({ kind: 'error', title: '文件过大', description: '单文件大小不能超过 200MB。' })
      return
    }

    try {
      if (isMockMode) {
        setUploadedFile({
          filename: file.name,
          durationSec: 12.34,
          sampleRate: 16000,
          channels: 1,
          bitDepth: 16,
          sizeBytes: file.size,
          format: file.name.split('.').pop()?.toUpperCase() ?? 'WAV',
          objectUrl: URL.createObjectURL(file),
          uploadedAt: new Date().toLocaleString('zh-CN', { hour12: false }),
          fingerprint: shortHash(`${file.name}-${file.size}`),
        })
        pushToast({
          kind: 'info',
          title: 'Mock 文件接入完成',
          description: 'Mock 模式下仅展示文件接入状态，分析结果来自固定演示数据。',
        })
      } else {
        const meta = await uploadFile(file)
        setUploadedFile({ ...meta, objectUrl: meta.audioUrl })
        pushToast({ kind: 'success', title: '文件上传完成', description: '后端已返回文件元数据。' })
      }
    } catch (error) {
      pushToast({ kind: 'error', title: '音频接入失败', description: error instanceof Error ? error.message : '请检查后端服务。' })
    }
  }

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      pushToast({ kind: 'error', title: '浏览器不支持录音', description: '请使用支持 MediaRecorder 的现代浏览器，并在 localhost 或 HTTPS 环境下测试。' })
      return
    }

    try {
      clearRecordedPreview()
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
        const file = new File([blob], name, { type })
        const objectUrl = URL.createObjectURL(blob)

        recordedUrlRef.current = objectUrl
        setRecordedUrl(objectUrl)
        setRecordedName(name)
        releaseRecorderStream()
        void handleFile(file)
      }

      recorder.start()
      setRecording(true)
      timerRef.current = window.setInterval(() => setRecordingSec((value) => value + 1), 1000)
      pushToast({ kind: 'success', title: '录音已开始', description: '请对着麦克风朗读测试语音。' })
    } catch (error) {
      releaseRecorderStream()
      pushToast({
        kind: 'error',
        title: '无法启动录音',
        description: error instanceof Error ? error.message : '请检查麦克风权限。',
      })
    }
  }

  const stopRecording = () => {
    if (!recorderRef.current || recorderRef.current.state === 'inactive') return
    clearRecordingTimer()
    recorderRef.current.stop()
    setRecording(false)
    pushToast({ kind: 'info', title: '录音已结束', description: '录音结果已作为当前音频输入。' })
  }

  const resetRecording = () => {
    if (recording) stopRecording()
    clearRecordingTimer()
    releaseRecorderStream()
    clearRecordedPreview()
    setRecordingSec(0)
    pushToast({ kind: 'info', title: '录音已重置' })
  }

  return (
    <Panel>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">音频接入</h2>
        <Badge tone={isMockMode ? 'cyan' : 'orange'}>{isMockMode ? 'Mock' : 'Backend'}</Badge>
      </div>

      <div className="mb-4 grid grid-cols-2 rounded-xl border border-white/10 bg-slate-950/50 p-1">
        <button
          type="button"
          className={`rounded-lg px-3 py-2 text-sm ${activeTab === 'upload' ? 'bg-cyan-400/15 text-cyan-100' : 'text-slate-400'}`}
          onClick={() => setActiveTab('upload')}
        >
          上传音频
        </button>
        <button
          type="button"
          className={`rounded-lg px-3 py-2 text-sm ${activeTab === 'record' ? 'bg-cyan-400/15 text-cyan-100' : 'text-slate-400'}`}
          onClick={() => setActiveTab('record')}
        >
          录音输入
        </button>
      </div>

      {activeTab === 'upload' ? (
        <div
          className="cursor-pointer rounded-2xl border border-dashed border-cyan-300/35 bg-cyan-300/5 p-6 text-center transition hover:bg-cyan-300/10"
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault()
            const file = event.dataTransfer.files.item(0)
            if (file) void handleFile(file)
          }}
        >
          <UploadCloud className="mx-auto h-10 w-10 text-cyan-200" />
          <p className="mt-3 text-sm font-semibold text-white">拖拽音频文件到此处，或点击上传</p>
          <p className="mt-1 text-xs text-slate-400">支持 .wav / .mp3 / .flac / .m4a，单文件 ≤ 200MB</p>
          <input
            ref={inputRef}
            type="file"
            accept=".wav,.mp3,.flac,.m4a,audio/*"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.item(0)
              if (file) void handleFile(file)
            }}
          />
        </div>
      ) : (
        <div className="rounded-2xl border border-cyan-300/25 bg-slate-950/50 p-5">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className={`grid h-11 w-11 place-items-center rounded-xl border ${recording ? 'border-red-300/40 bg-red-400/15 text-red-100' : 'border-cyan-300/30 bg-cyan-300/10 text-cyan-100'}`}>
                <Mic className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-semibold text-white">系统麦克风录音</p>
                <p className="text-xs text-slate-400">{recording ? '录音中，请保持环境安静' : '录音后将自动作为当前输入音频'}</p>
              </div>
            </div>
            <Badge tone={recording ? 'red' : recordedUrl ? 'green' : 'slate'}>{recording ? '录音中' : recordedUrl ? '已录制' : '待开始'}</Badge>
          </div>

          <div className="mt-5 rounded-xl border border-white/10 bg-slate-950/60 p-4 text-center">
            <p className="font-mono text-3xl font-bold tracking-[0.12em] text-cyan-100">{formatDuration(recordingSec)}</p>
            <p className="mt-1 text-xs text-slate-500">录音时长</p>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            {!recording ? (
              <Button type="button" icon={<Mic className="h-4 w-4" />} onClick={() => void startRecording()}>
                开始录音
              </Button>
            ) : (
              <Button type="button" variant="danger" icon={<StopCircle className="h-4 w-4" />} onClick={stopRecording}>
                停止录音
              </Button>
            )}
            <Button type="button" variant="secondary" icon={<RotateCcw className="h-4 w-4" />} onClick={resetRecording}>
              重录
            </Button>
          </div>

          {recordedUrl ? (
            <div className="mt-4 rounded-xl border border-emerald-300/20 bg-emerald-300/8 p-3">
              <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-emerald-50">
                <PlayCircle className="h-4 w-4" />
                {recordedName}
              </div>
              <audio className="w-full" controls src={recordedUrl} />
            </div>
          ) : null}

          <p className="mt-4 rounded-lg border border-cyan-300/20 bg-cyan-300/8 p-3 text-xs leading-5 text-cyan-100">
            录音输入会调用浏览器麦克风权限。Backend 模式下，停止录音后会把录音文件上传后端；Mock 模式下，仅作为演示输入展示。
          </p>
        </div>
      )}

      {uploadedFile ? (
        <div className="mt-5 rounded-2xl border border-white/10 bg-slate-950/45 p-4">
          <div className="mb-3 flex items-center gap-2">
            <FileAudio className="h-5 w-5 text-cyan-200" />
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-white">{uploadedFile.filename}</p>
              <p className="text-xs text-slate-400">{uploadedFile.fingerprint}</p>
            </div>
          </div>
          <AudioWaveform dense />
          <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-slate-300">
            <span>时长：{uploadedFile.durationSec}s</span>
            <span>采样率：{uploadedFile.sampleRate / 1000}kHz</span>
            <span>声道：{uploadedFile.channels}</span>
            <span>位深：{uploadedFile.bitDepth} bit</span>
            <span>大小：{formatBytes(uploadedFile.sizeBytes)}</span>
            <span>上传：{uploadedFile.uploadedAt}</span>
          </div>

          {isMockMode ? (
            <p className="mt-3 rounded-lg border border-cyan-300/20 bg-cyan-300/8 p-3 text-xs leading-5 text-cyan-100">
              Mock 模式下仅展示文件接入状态，分析结果来自固定演示数据。
            </p>
          ) : null}
          {uploadedFile.objectUrl ? (
            <div className="mt-4 rounded-xl border border-cyan-300/20 bg-slate-950/45 p-3">
              <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-cyan-100">
                <FileAudio className="h-4 w-4" />
                播放预览
              </div>
              <audio className="w-full" controls src={uploadedFile.objectUrl} />
            </div>
          ) : (
            <Button variant="secondary" className="mt-4 w-full" icon={<FileAudio className="h-4 w-4" />}>
              播放预览
            </Button>
          )}
        </div>
      ) : null}
    </Panel>
  )
}
