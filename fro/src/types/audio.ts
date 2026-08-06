export interface AudioFileMeta {
  fileId?: string
  filename: string
  durationSec?: number
  duration?: number
  sampleRate?: number
  channels?: number
  bitDepth?: number
  codec?: string
  metadataStatus?: 'available' | 'partial' | 'unavailable'
  metadataSource?: string
  metadataReason?: string
  sizeBytes: number
  format: string
  src?: string
  audioUrl?: string
  objectUrl?: string
  downloadUrl?: string
  uploadedAt?: string
  fingerprint?: string
}
