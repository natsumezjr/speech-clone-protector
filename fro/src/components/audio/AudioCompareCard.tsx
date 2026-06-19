import { FileAudio } from 'lucide-react'
import { Panel } from '@/components/common/Panel'
import type { AudioFileMeta } from '@/types/audio'
import { formatBytes } from '@/utils/format'
import { AudioWaveform } from './AudioWaveform'
import { MockAudioPlayer } from './MockAudioPlayer'

interface AudioCompareCardProps {
  title: string
  description: string
  audio: AudioFileMeta
  variant?: 'cyan' | 'green' | 'orange'
}

export function AudioCompareCard({ title, description, audio, variant = 'cyan' }: AudioCompareCardProps) {
  return (
    <Panel>
      <div className="mb-4 flex items-start gap-3">
        <div className="rounded-xl border border-cyan-300/20 bg-cyan-300/10 p-2 text-cyan-100">
          <FileAudio className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-white">{title}</h3>
          <p className="truncate text-sm text-slate-400">{audio.filename}</p>
        </div>
      </div>
      <AudioWaveform dense variant={variant} />
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-slate-300 md:grid-cols-4">
        <span>时长 {audio.durationSec}s</span>
        <span>采样率 {audio.sampleRate / 1000}kHz</span>
        <span>格式 {audio.format}</span>
        <span>大小 {formatBytes(audio.sizeBytes)}</span>
      </div>
      <p className="mt-4 text-sm leading-6 text-slate-300">{description}</p>
      <div className="mt-4">
        <MockAudioPlayer />
      </div>
    </Panel>
  )
}
