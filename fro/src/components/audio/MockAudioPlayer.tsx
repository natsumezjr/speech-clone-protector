import { Play, Pause } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import { Button } from '@/components/common/Button'
import { createMockProtectedWavBlob } from '@/utils/mockWav'

export function MockAudioPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const src = useMemo(() => URL.createObjectURL(createMockProtectedWavBlob()), [])

  const toggle = async () => {
    if (!audioRef.current) return
    if (playing) {
      audioRef.current.pause()
      setPlaying(false)
      return
    }
    await audioRef.current.play()
    setPlaying(true)
  }

  return (
    <div className="flex items-center gap-3">
      <Button variant="secondary" icon={playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />} onClick={toggle}>
        {playing ? '暂停' : '播放'}
      </Button>
      <audio ref={audioRef} src={src} onEnded={() => setPlaying(false)} />
      <span className="text-xs text-slate-400">Mock WAV 试听</span>
    </div>
  )
}
