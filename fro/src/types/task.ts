import type { DataMode } from '@/config/runtime'
import type { AudioFileMeta } from './audio'

export type TaskStatus = 'queued' | 'running' | 'completed' | 'success' | 'failed' | 'error'
export type ProtectionMode = 'standard' | 'strong' | 'high_fidelity' | 'custom' | 'joint'
export type ProtectionTarget = 'semantic' | 'timbre'
export type TaskStage =
  | 'file_preprocess'
  | 'encoder_loading'
  | 'perturbation_optimization'
  | 'psychoacoustic_constraint'
  | 'result_evaluation'
  | 'report_generation'

export interface ProtectionTaskRequest {
  fileId?: string
  mode: Exclude<ProtectionMode, 'joint'>
  targets: ProtectionTarget[]
  semantic: {
    enabled: boolean
    asrModel: string
    encoders: string[]
    lambdaSemantic: number
  }
  timbre: {
    enabled: boolean
    mode: 'untargeted' | 'targeted'
    encoders: string[]
    lambdaTimbre: number
  }
  psychoacoustic: {
    enabled: boolean
    lambdaPsy: number
  }
  optimization: {
    epsilon: number
    steps: number
  }
}

export interface TaskStatusResponse {
  taskId: string
  status: TaskStatus
  progress: number
  stage: TaskStage
  message: string
  createdAt: string
  updatedAt: string
  error: string | null
}

export interface AsrMetrics {
  originalText: string
  protectedText: string
  wer?: number
  cer?: number
  tokenChangeRate?: number
  tokenErrorRate?: number
  semanticDrift?: number
  insertRate?: number
  deleteRate?: number
}

export interface SpeakerMetrics {
  simBefore: number
  simAfter: number
  simDropRate: number
  embeddingDistanceBefore: number
  embeddingDistanceAfter: number
}

export interface QualityMetrics {
  snr: number
  pesq: number
  mosLqo: number
}

export interface TrendPoint {
  step: number
  wer: number
  sim: number
  mos: number
  pesq: number
  elapsed: number
}

export interface PsychoacousticPoint {
  frequency: number
  maskingThreshold: number
  perturbation: number
}

export interface TaskResult {
  taskId: string
  status: TaskStatus
  mode: ProtectionMode
  dataMode: DataMode
  verdict: string
  score: number
  createdAt?: string
  submittedAt?: string
  completedAt: string
  elapsedSec: number
  inputSource?: string
  language?: string
  processingModel?: string
  optimizationTarget?: string
  asrModel?: string
  artifacts?: Array<{
    label: string
    filename: string
    sizeBytes?: number
    sizeLabel?: string
  }>
  originalAudio: AudioFileMeta
  protectedAudio: AudioFileMeta
  cloneResults?: CloneVoiceResult[]
  asr: AsrMetrics
  speaker: SpeakerMetrics
  quality: QualityMetrics
  charts: {
    psychoacoustic: PsychoacousticPoint[]
    trend: TrendPoint[]
    radarBefore: number[]
    radarAfter: number[]
  }
}

export interface CloneVoiceRequest {
  text: string
  model: string
  language?: string
  speed?: number
  speakerPrompt?: string
}

export interface CloneVoiceResult {
  cloneId: string
  taskId: string
  status: 'queued' | 'running' | 'completed' | 'success' | 'partial' | 'failed' | 'error'
  source?: string
  message?: string
  request: CloneVoiceRequest
  originalCloneAudio: AudioFileMeta
  protectedCloneAudio: AudioFileMeta
}

export interface HistoryTask {
  taskId: string
  filename: string
  protectedFilename: string
  mode: ProtectionMode
  dataMode: DataMode
  status: TaskStatus
  wer: number
  simDropRate: number
  pesq: number
  createdAt: string
}

export interface UploadedFileState extends AudioFileMeta {
  objectUrl?: string
  rawFile?: File
}
