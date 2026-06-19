export interface AudioFileMeta {
  fileId?: string
  filename: string
  durationSec: number
  sampleRate: number
  channels: number
  bitDepth: number
  sizeBytes: number
  format: string
  audioUrl?: string
  uploadedAt?: string
  fingerprint?: string
}
