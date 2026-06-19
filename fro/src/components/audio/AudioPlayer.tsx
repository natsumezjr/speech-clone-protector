import { Download, Pause, Play } from 'lucide-react'
import { useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { formatDurationSeconds } from '@/utils/audio'

interface AudioPlayerProps {
  src?: string
  title?: string
  filename?: string
  disabledReason?: string
  downloadable?: boolean
  downloadFilename?: string
  onDownload?: () => void
  onPlayRequest?: () => Promise<string | undefined> | string | undefined
}

export function AudioPlayer({
  src,
  title = '音频试听',
  filename,
  disabledReason = '暂无可播放音频源',
  downloadable = false,
  downloadFilename,
  onDownload,
  onPlayRequest,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const [resolvedSrc, setResolvedSrc] = useState(src)
  const [loading, setLoading] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const currentSrc = src ?? resolvedSrc
  const disabled = !currentSrc && !onPlayRequest

  const toggle = async () => {
    if (playing) {
      audioRef.current?.pause()
      setPlaying(false)
      return
    }

    let nextSrc = currentSrc
    if (!nextSrc && onPlayRequest) {
      setLoading(true)
      try {
        nextSrc = await onPlayRequest()
        setResolvedSrc(nextSrc)
      } finally {
        setLoading(false)
      }
    }

    if (!nextSrc) return

    if (audioRef.current) {
      audioRef.current.src = nextSrc
      await audioRef.current.play()
      setPlaying(true)
    }
  }

  const handleSeek = (value: number) => {
    if (!audioRef.current || !Number.isFinite(duration)) return
    audioRef.current.currentTime = value
    setCurrentTime(value)
  }

  const playableDuration = Number.isFinite(duration) ? duration : 0
  const progress = playableDuration > 0 ? (currentTime / playableDuration) * 100 : 0

  return (
    <div className="rounded-[8px] border border-cyan-300/12 bg-slate-950/18 p-3">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="secondary"
          icon={playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          onClick={() => void toggle()}
          disabled={disabled || loading}
        >
          {loading ? '加载中' : playing ? '暂停' : '播放'}
        </Button>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-semibold text-slate-200">{title}</p>
          <p className="truncate text-xs text-slate-500">{currentSrc ? filename : disabledReason}</p>
        </div>
        {downloadable ? (
          onDownload ? (
            <Button variant="ghost" className="h-9 px-2" title="下载音频" onClick={onDownload}>
              <Download className="h-4 w-4" />
            </Button>
          ) : currentSrc ? (
            <a
              className="inline-flex h-9 items-center justify-center rounded-lg px-2 text-slate-200 transition hover:bg-white/[0.07]"
              href={currentSrc}
              download={downloadFilename ?? filename}
              title="下载音频"
            >
              <Download className="h-4 w-4" />
            </a>
          ) : null
        ) : null}
      </div>
      <div className="mt-3 grid grid-cols-[44px_minmax(0,1fr)_44px] items-center gap-3">
        <span className="font-mono text-[10px] text-slate-500">{formatDurationSeconds(currentTime)}</span>
        <div className="relative h-5">
          <div className="absolute left-0 right-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-slate-800">
            <div className="h-full rounded-full bg-cyan-400" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
          </div>
          <input
            className="absolute inset-0 h-5 w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
            type="range"
            min={0}
            max={playableDuration || 0}
            step={0.01}
            value={Math.min(currentTime, playableDuration || currentTime)}
            disabled={!currentSrc || playableDuration <= 0}
            aria-label={`${title} 播放进度`}
            onChange={(event) => handleSeek(Number(event.target.value))}
          />
        </div>
        <span className="text-right font-mono text-[10px] text-slate-500">{formatDurationSeconds(playableDuration || undefined)}</span>
      </div>
      <audio
        ref={audioRef}
        src={currentSrc}
        onLoadStart={() => {
          setPlaying(false)
          setCurrentTime(0)
          setDuration(0)
        }}
        onLoadedMetadata={(event) => {
          setDuration(event.currentTarget.duration)
          setCurrentTime(event.currentTarget.currentTime)
        }}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        onEnded={() => {
          setPlaying(false)
          setCurrentTime(0)
        }}
        onPause={() => setPlaying(false)}
        preload="metadata"
      />
    </div>
  )
}
