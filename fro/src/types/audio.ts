export interface AudioFileMeta {
  fileId?: string
  filename: string
  durationSec?: number
  duration?: number
  sampleRate?: number
  channels?: number
  bitDepth?: number
  sizeBytes: number
  format: string
  src?: string
  audioUrl?: string
  objectUrl?: string
  downloadUrl?: string
  uploadedAt?: string
  fingerprint?: string
}
