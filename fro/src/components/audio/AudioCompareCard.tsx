import { FileAudio } from 'lucide-react'
import { Panel } from '@/components/common/Panel'
import type { AudioFileMeta } from '@/types/audio'
import { getAudioDuration, getAudioSource } from '@/utils/audio'
import { formatBytes } from '@/utils/format'
import { AudioWaveform } from './AudioWaveform'
import { AudioPlayer } from './AudioPlayer'

interface AudioCompareCardProps {
  title: string
  description: string
  audio: AudioFileMeta
  variant?: 'cyan' | 'green' | 'orange'
}

export function AudioCompareCard({ title, description, audio, variant = 'cyan' }: AudioCompareCardProps) {
  const src = getAudioSource(audio)
  const duration = getAudioDuration(audio)

  return (
    <Panel className="border-sky-400/22 bg-[#071226]/88 p-6">
      <div className="mb-5 flex items-start gap-3">
        <div className="rounded-xl border border-cyan-300/24 bg-cyan-300/10 p-3 text-cyan-100">
          <FileAudio className="h-6 w-6" />
        </div>
        <div className="min-w-0">
          <h3 className="text-xl font-bold text-white">{title}</h3>
          <p className="mt-1 truncate text-base font-semibold text-slate-400">{audio.filename}</p>
        </div>
      </div>
      <AudioWaveform dense variant={variant} />
      <div className="mt-5 grid grid-cols-2 gap-3 text-sm text-slate-300 md:grid-cols-4">
        <span>时长 {duration ? `${duration.toFixed(2)}s` : '待解析'}</span>
        <span>采样率 {audio.sampleRate ? `${audio.sampleRate / 1000}kHz` : '待后端解析'}</span>
        <span>格式 {audio.format}</span>
        <span>大小 {formatBytes(audio.sizeBytes)}</span>
      </div>
      <p className="mt-5 text-base leading-7 text-slate-300">{description}</p>
      <div className="mt-5">
        <AudioPlayer
          src={src}
          title={title}
          filename={audio.filename}
          downloadable={Boolean(src)}
          downloadFilename={audio.filename}
          disabledReason="该音频暂未返回可播放 URL"
        />
      </div>
    </Panel>
  )
}
