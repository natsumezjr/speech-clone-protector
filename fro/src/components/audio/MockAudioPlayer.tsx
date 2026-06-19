import { useEffect, useMemo } from 'react'
import { AudioPlayer } from './AudioPlayer'
import { createMockProtectedWavBlob } from '@/utils/mockWav'

/**
 * @deprecated Use AudioPlayer with an explicit src. This wrapper only exists for
 * legacy demo surfaces that still need a generated mock WAV.
 */
export function MockAudioPlayer() {
  const src = useMemo(() => URL.createObjectURL(createMockProtectedWavBlob()), [])

  useEffect(() => {
    return () => URL.revokeObjectURL(src)
  }, [src])

  return <AudioPlayer src={src} title="Mock WAV 试听" filename="mock-protected.wav" />
}
