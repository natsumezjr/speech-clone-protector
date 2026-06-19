import type { AudioFileMeta } from '@/types/audio'

export function getAudioSource(audio?: Pick<AudioFileMeta, 'src' | 'objectUrl' | 'audioUrl' | 'downloadUrl'> | null) {
  return audio?.src ?? audio?.objectUrl ?? audio?.audioUrl ?? audio?.downloadUrl
}

export function getAudioDuration(audio?: Pick<AudioFileMeta, 'durationSec' | 'duration'> | null) {
  return audio?.durationSec ?? audio?.duration
}

export function formatDurationSeconds(totalSeconds?: number) {
  if (typeof totalSeconds !== 'number' || !Number.isFinite(totalSeconds)) return '待解析'
  const rounded = Math.max(0, Math.round(totalSeconds))
  const minutes = Math.floor(rounded / 60)
  const seconds = rounded % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

export function readAudioDuration(src: string) {
  return new Promise<number>((resolve, reject) => {
    const audio = new Audio()
    audio.preload = 'metadata'

    const cleanup = () => {
      audio.removeAttribute('src')
      audio.load()
    }

    audio.onloadedmetadata = () => {
      const duration = audio.duration
      cleanup()
      if (Number.isFinite(duration)) {
        resolve(duration)
      } else {
        reject(new Error('浏览器暂无法解析音频时长。'))
      }
    }

    audio.onerror = () => {
      cleanup()
      reject(new Error('浏览器无法读取该音频文件。'))
    }

    audio.src = src
  })
}

export function getRecorderMimeType() {
  if (typeof MediaRecorder === 'undefined') return undefined

  return ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/mpeg'].find((type) => MediaRecorder.isTypeSupported(type))
}

export function getRecordedExtension(mimeType: string) {
  if (mimeType.includes('mp4')) return 'm4a'
  if (mimeType.includes('mpeg')) return 'mp3'
  return 'webm'
}
