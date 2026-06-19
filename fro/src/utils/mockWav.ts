export function createMockProtectedWavBlob(durationSec = 4, sampleRate = 16000) {
  const channels = 1
  const bitsPerSample = 16
  const bytesPerSample = bitsPerSample / 8
  const samples = Math.floor(durationSec * sampleRate)
  const dataSize = samples * channels * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(view, 8, 'WAVE')
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * channels * bytesPerSample, true)
  view.setUint16(32, channels * bytesPerSample, true)
  view.setUint16(34, bitsPerSample, true)
  writeString(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  for (let index = 0; index < samples; index += 1) {
    const t = index / sampleRate
    const envelope = 0.35 + 0.15 * Math.sin(2 * Math.PI * 1.7 * t)
    const carrier = Math.sin(2 * Math.PI * 226 * t) + 0.42 * Math.sin(2 * Math.PI * 341 * t)
    const shimmer = 0.08 * Math.sin(2 * Math.PI * 17 * t)
    const value = Math.max(-1, Math.min(1, (carrier * envelope + shimmer) * 0.18))
    view.setInt16(44 + index * 2, value * 0x7fff, true)
  }

  return new Blob([view], { type: 'audio/wav' })
}

function writeString(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index))
  }
}
