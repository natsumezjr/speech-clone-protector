import { FileAudio, Mic, UploadCloud } from 'lucide-react'
import { useRef, useState } from 'react'
import { Badge } from '@/components/common/Badge'
import { Button } from '@/components/common/Button'
import { Panel } from '@/components/common/Panel'
import { isMockMode } from '@/config/runtime'
import { uploadFile } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import { formatBytes, shortHash } from '@/utils/format'
import { AudioWaveform } from './AudioWaveform'

export function AudioUploadPanel() {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [activeTab, setActiveTab] = useState<'upload' | 'record'>('upload')
  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const setUploadedFile = useTaskStore((state) => state.setUploadedFile)
  const pushToast = useAppStore((state) => state.pushToast)

  const handleFile = async (file: File) => {
    if (file.size > 200 * 1024 * 1024) {
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

  return (
    <Panel>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">音频接入</h2>
        <Badge tone={isMockMode ? 'cyan' : 'orange'}>{isMockMode ? 'Mock' : 'Backend'}</Badge>
      </div>
      <div className="mb-4 grid grid-cols-2 rounded-xl border border-white/10 bg-slate-950/50 p-1">
        <button
          className={`rounded-lg px-3 py-2 text-sm ${activeTab === 'upload' ? 'bg-cyan-400/15 text-cyan-100' : 'text-slate-400'}`}
          onClick={() => setActiveTab('upload')}
        >
          上传音频
        </button>
        <button
          className={`rounded-lg px-3 py-2 text-sm ${activeTab === 'record' ? 'bg-cyan-400/15 text-cyan-100' : 'text-slate-400'}`}
          onClick={() => {
            setActiveTab('record')
            pushToast({ kind: 'info', title: '录音输入接口预留' })
          }}
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
        <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-6 text-center">
          <Mic className="mx-auto h-10 w-10 text-slate-300" />
          <p className="mt-3 text-sm font-semibold text-white">录音输入接口预留</p>
          <p className="mt-1 text-xs text-slate-400">后续可接入浏览器录音与实时上传能力。</p>
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
          <Button variant="secondary" className="mt-4 w-full" icon={<FileAudio className="h-4 w-4" />}>
            播放预览
          </Button>
        </div>
      ) : null}
    </Panel>
  )
}
