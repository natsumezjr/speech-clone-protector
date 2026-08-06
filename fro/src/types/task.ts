import type { AudioFileMeta } from './audio'

export type DataMode = 'backend'

export type TaskStatus = 'queued' | 'running' | 'completed' | 'success' | 'failed' | 'error' | 'cancelled'
export type ProtectionMode = 'standard' | 'strong' | 'high_fidelity' | 'custom' | 'joint'
export type ProtectionTarget = 'semantic' | 'timbre'
export type TaskStage =
  | 'file_preprocess'
  | 'encoder_loading'
  | 'perturbation_optimization'
  | 'psychoacoustic_constraint'
  | 'result_evaluation'
  | 'report_generation'
  | 'protect_generation'
  | 'semantic_tokenizer_eval'
  | 'asr_eval'
  | 'speaker_eval'
  | 'downstream_tts_eval'
  | 'perception_eval'

export interface ProtectionTaskRequest {
  fileId?: string
  mode: Exclude<ProtectionMode, 'joint'>
  profile?: 'formal' | string
  targets: ProtectionTarget[]
  semantic: {
    enabled: boolean
    asrModel?: string
    asrModels?: string[]
    encoders: string[]
    tokenizerPath?: string
    hubertPath?: string
    whisperPath?: string
    weightSemantic?: number
    lambdaSemantic?: number
  }
  timbre: {
    enabled: boolean
    mode: 'untargeted' | 'targeted'
    encoders: string[]
    weightIdentity?: number
    lambdaId?: number
    weightFeature?: number
    lambdaTimbre?: number
  }
  psychoacoustic: {
    enabled: boolean
    weightPsy?: number
    lambdaPsy?: number
  }
  optimization: {
    epsilon: number
    steps: number
    weightL2?: number
    lambdaL2?: number
  }
}

export interface AsrMetrics {
  model?: string
  asrModel?: string | null
  language?: string | null
  referenceText?: string | null
  originalText: string | null
  protectedText: string | null
  wer?: number | null
  cer?: number | null
  tokenChangeRate?: number | null
  tokenErrorRate?: number | null
  semanticDrift?: number | null
  insertRate?: number | null
  deleteRate?: number | null
  substituteRate?: number | null
  editCounts?: {
    level: 'word' | 'char' | string
    referenceLength: number
    substitutions: number
    insertions: number
    deletions: number
    totalErrors: number
  } | null
  errorShares?: {
    substituteShare?: number | null
    insertShare?: number | null
    deleteShare?: number | null
  } | null
  metricLevel?: 'word' | 'char' | string | null
  asrProtectionScore?: number | null
  diffOps?: DiffOp[] | null
  trend?: Array<Record<string, number>> | null
  createdAt?: string | null
  status?: string
  error?: string | null
  reason?: string | null
}

export interface AsrEvalResponse {
  taskId: string
  status: string
  asr?: AsrMetrics
  asrSubId?: string
}

export interface SubtaskStatusSnapshot {
  status?: TaskStatus | string
  progress?: number | null
  stage?: TaskStage | string | null
  message?: string | null
  elapsedSec?: number | null
  error?: string | ApiErrorPayload | null
  createdAt?: string | null
  updatedAt?: string | null
  asrSubId?: string | null
  cloneSubId?: string | null
  asrResult?: AsrEvalResponse | null
  cloneResult?: CloneVoiceResult | null
}

export interface TaskStatusResponse {
  taskId: string
  status: TaskStatus
  progress: number
  stage: TaskStage
  message: string
  createdAt: string
  updatedAt: string
  currentStep?: number | null
  totalSteps?: number | null
  elapsedSec?: number | null
  error: string | ApiErrorPayload | null
  asrResult?: AsrEvalResponse | null
  cloneResult?: CloneVoiceResult | null
  asrTask?: SubtaskStatusSnapshot | null
  cloneTask?: SubtaskStatusSnapshot | null
}

export interface SpeakerMetrics {
  simBefore: number | null
  simAfter: number | null
  simDropRate: number | null
  embeddingDistanceBefore: number | null
  embeddingDistanceAfter: number | null
  simOriginalProtected?: number | null
  embeddingDistance?: number | null
  source?: string
  status?: string
}

export interface QualityMetrics {
  snr: number | null
  pesq: number | null
  mosLqo: number | null
  l2Norm?: number | null
  psychoacousticViolationRate?: number | null
  status?: string
}

export interface TrendPoint {
  step: number
  wer?: number | null
  sim?: number | null
  mos?: number | null
  pesq?: number | null
  elapsed?: number | null
}

export interface LossTrendPoint {
  step: number
  Lid?: number | null
  Lfeat?: number | null
  Lsem: number | null
  Lpsy: number | null
  L2: number | null
  total?: number | null
  snr?: number | null
  stepElapsedSec?: number | null
}

export interface LossFinal {
  Lid?: number | null
  Lfeat?: number | null
  Lsem: number | null
  Lpsy: number | null
  L2: number | null
  total?: number | null
  snr?: number | null
}

export interface PsychoacousticPoint {
  frequency: number
  maskingThreshold: number
  perturbation?: number
  perturbationPsd?: number
}

export type PsychoacousticSliceMode = 'mean' | 'frame'

export interface PsychoacousticSliceResponse {
  mode: PsychoacousticSliceMode
  requestedTimeSec?: number | null
  actualTimeSec?: number | null
  frameIndex?: number | null
  frameCount?: number | null
  sampleRate?: number | null
  hopLength?: number | null
  nFft?: number | null
  aggregation?: 'time_mean' | 'single_frame'
  lPsy?: number | null
  overMaskRate?: number | null
  maskingThreshold: Array<{ frequencyHz: number; thresholdDb: number }>
  perturbationSpectrum: Array<{ frequencyHz: number; powerDb: number }>
  charts?: {
    psychoacoustic?: Array<{
      frequency: number
      maskingThreshold: number
      perturbation?: number
      perturbationPsd?: number
    }>
  }
  metricSources?: Record<string, MetricSource>
}

export type DiffOp =
  | { type: 'equal' | 'insert' | 'delete'; text: string }
  | { type: 'replace'; from: string; to: string }

export interface PerturbationMetrics {
  l2Norm?: number | null
  l2Rms?: number | null
  linfNorm?: number | null
  epsilon?: number | null
  epsilonNorm?: 'l2' | 'linf' | string | null
  epsilonUsageRate?: number | null
  snr?: number | null
  clippingRate?: number | null
}

export interface ProtectionQuality {
  snr?: number | null
  pesq?: number | null
  stoi?: number | null
  mos?: number | null
  mosLqo?: number | null
  qualityLevel?: string | null
}

export interface PsychoacousticMetrics {
  lPsy?: number | null
  overMaskRate?: number | null
  frameCount?: number | null
  sampleRate?: number | null
  hopLength?: number | null
  nFft?: number | null
  aggregation?: 'time_mean' | 'single_frame' | string | null
  maskingThreshold?: Array<{ frequencyHz: number; thresholdDb: number }> | null
  perturbationSpectrum?: Array<{ frequencyHz: number; powerDb: number }> | null
}

export interface LossWeights {
  lambdaId?: number | null
  lambdaFeat?: number | null
  lambdaSem?: number | null
  lambdaPsy?: number | null
  lambda2?: number | null
}

export type AsrEval = AsrMetrics

export interface RadarPoint {
  name: string
  value: number | null
  status?: 'available' | 'unavailable' | 'partial' | 'not_run' | 'error' | string
  reason?: string | null
  formula?: string | null
  rawMetricKeys?: string[] | null
}

export interface CloneEval {
  cloneModel?: string | null
  speakerEvalModel?: string | null
  speakerModel?: string | null
  targetText?: string | null
  originalCloneAudio?: AudioFileMeta | null
  protectedCloneAudio?: AudioFileMeta | null
  originalSimilarity?: number | null
  protectedSimilarity?: number | null
  similarityDropRate?: number | null
  embeddingDistanceBefore?: number | null
  embeddingDistanceAfter?: number | null
  embeddingDistanceIncreaseRate?: number | null
  cloneConfidenceBefore?: number | null
  cloneConfidenceAfter?: number | null
  cloneConfidenceDropRate?: number | null
  cloneRadar?: RadarPoint[] | null
  cloneTrend?: Array<Record<string, number>> | null
  cloneDefenseScore?: number | null
  createdAt?: string | null
  status?: string | null
  reason?: string | null
}

export interface MetricSource {
  source?: string
  status?: string
  reason?: string
  formula?: string
  metric?: string
}

export interface ApiErrorPayload {
  code?: string
  message: string
  requestId?: string
  taskId?: string
  stage?: string
  details?: unknown
}

export interface CapabilityChain {
  status: 'available' | 'unavailable' | 'partial' | string
  reason?: string | null
  available?: string[]
  unavailable?: string[]
}

export interface NumericConfigRange {
  min: number
  max: number
  step: number
}

export interface RuntimeModelOption {
  label?: string
  value: string
  backendValue?: string
  branch?: string
  backend?: string
  defaultPath?: string
  status?: string
  reason?: string | null
  languages?: string[]
}

export interface RuntimeChoiceOption {
  label: string
  value: string
  description?: string
  targetPolicy?: 'selectable' | 'joint_only' | string
}

export interface RuntimeModePreset {
  semantic?: Partial<ProtectionTaskRequest['semantic']>
  timbre?: Partial<ProtectionTaskRequest['timbre']>
  psychoacoustic?: Partial<ProtectionTaskRequest['psychoacoustic']>
  optimization?: Partial<ProtectionTaskRequest['optimization']>
}

export interface ProtectionRuntimeConfig {
  defaults: ProtectionTaskRequest
  profiles?: Record<string, ProtectionTaskRequest>
  activeDefaultProfile?: string
  ranges: Record<string, NumericConfigRange>
  models: {
    semantic?: Array<string | RuntimeModelOption>
    asr?: Array<string | RuntimeModelOption>
    feature?: Array<string | RuntimeModelOption>
    timbre?: Array<string | RuntimeModelOption>
    tts?: Array<string | RuntimeModelOption>
    [key: string]: Array<string | RuntimeModelOption> | undefined
  }
  formSchema?: RuntimeFormSchema
  constraints?: {
    maxAudioSizeBytes?: number
    [key: string]: unknown
  }
  clone?: {
    defaults?: {
      model?: string
      backendValue?: string
      language?: string
      uiPreferredLanguage?: string
      speed?: number
    }
    languages?: string[]
    speeds?: number[]
  }
  modes?: RuntimeChoiceOption[]
  targets?: RuntimeChoiceOption[]
  modePresets?: Record<string, RuntimeModePreset>
}

export interface RuntimeFormField {
  label: string
  path: string
  default: number
  min: number
  max: number
  step: number
  unit?: string
  description?: string
}

export interface RuntimeFormSchema {
  defaults?: {
    formal?: ProtectionTaskRequest
  }
  activeDefaultProfile?: string
  profiles?: RuntimeChoiceOption[]
  fields?: Record<string, RuntimeFormField>
  modelOptions?: {
    semanticEncoders?: Array<string | RuntimeModelOption>
    timbreEncoders?: Array<string | RuntimeModelOption>
    asrModels?: Array<string | RuntimeModelOption>
    ttsModels?: Array<string | RuntimeModelOption>
  }
}

export interface CapabilitiesResponse {
  ok: boolean
  device?: string
  chains?: Record<string, CapabilityChain>
  config?: ProtectionRuntimeConfig
  defaults?: ProtectionRuntimeConfig['defaults']
  ranges?: ProtectionRuntimeConfig['ranges']
  models?: ProtectionRuntimeConfig['models']
  constraints?: ProtectionRuntimeConfig['constraints']
  cache?: {
    hit: boolean
    revision: number
    refreshedAt?: string | null
    refreshRequested?: boolean
    refreshing?: boolean
    strategy?: string
  }
}

export interface TaskResult {
  taskId: string
  status: TaskStatus
  mode: ProtectionMode
  dataMode: DataMode
  verdict: string
  score: number | null
  createdAt?: string
  submittedAt?: string
  completedAt?: string
  elapsedSec?: number | null
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
  perturbation?: PerturbationMetrics | null
  protectionQuality?: ProtectionQuality | null
  psychoacoustic?: PsychoacousticMetrics | null
  lossFinal?: LossFinal | null
  lossWeights?: LossWeights | null
  optimizationTrace?: LossTrendPoint[] | null
  averageStepSec?: number | null
  selectedStep?: number | null
  effectiveConfig?: Record<string, unknown> | null
  presetName?: string | null
  asrEval?: AsrEval | null
  cloneEval?: CloneEval | null
  cloneResults?: CloneVoiceResult[]
  speakerFeatureMap?: {
    radar?: RadarPoint[] | null
  } | null
  asr: AsrMetrics
  speaker: SpeakerMetrics
  quality: QualityMetrics
  metricSources?: Record<string, MetricSource>
  generation?: {
    lossFinal?: LossFinal
    lossWeights?: LossWeights
    optimizationTrace?: LossTrendPoint[]
    steps?: number | null
    maxSteps?: number | null
    selectedStep?: number | null
    snrDb?: number | null
    presetName?: string | null
    effectiveConfig?: Record<string, unknown> | null
    realProtect?: boolean | null
    source?: string
    status?: string
    mode?: string
  }
  raw?: unknown
  charts: {
    psychoacoustic: PsychoacousticPoint[]
    trend: TrendPoint[]
    optimizationTrend: LossTrendPoint[]
    radarBefore?: number[]
    radarAfter?: number[]
    chainRadar?: Array<{ name: string; value: number | null; status?: string }>
    speakerRadar?: RadarPoint[] | null
  }
}

export interface TaskDetailsResponse {
  taskId: string
  status: TaskStatus
  details: Record<string, unknown>
  summary?: Record<string, unknown>
  chains?: Array<Record<string, unknown>>
  charts?: Record<string, unknown>
  raw?: unknown
}

export interface CloneVoiceRequest {
  text: string
  model: string
  language?: string
  speed?: number
  speakerPrompt?: string
}

export interface AsrEvalRequest {
  model: string
  referenceText?: string
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
  cloneEval?: CloneEval | null
}

export interface HistoryTask {
  taskId: string
  filename: string
  protectedFilename: string
  mode: ProtectionMode
  targetMode?: 'semantic' | 'timbre' | 'joint'
  parameters?: {
    weightSemantic?: number | null
    weightIdentity?: number | null
    weightFeature?: number | null
    weightPsy?: number | null
    weightL2?: number | null
  }
  dataMode: DataMode
  status: TaskStatus
  progress?: number | null
  stage?: TaskStage | string | null
  message?: string | null
  protectionStatus?: TaskStatus | string | null
  protectionProgress?: number | null
  protectionStage?: TaskStage | string | null
  protectionMessage?: string | null
  protectionElapsedSec?: number | null
  protectionError?: string | ApiErrorPayload | null
  asrStatus?: TaskStatus | string | null
  asrProgress?: number | null
  asrStage?: TaskStage | string | null
  asrMessage?: string | null
  asrElapsedSec?: number | null
  asrStartedAt?: string | null
  asrCompletedAt?: string | null
  asrError?: string | ApiErrorPayload | null
  cloneStatus?: TaskStatus | string | null
  cloneProgress?: number | null
  cloneStage?: TaskStage | string | null
  cloneMessage?: string | null
  cloneElapsedSec?: number | null
  cloneStartedAt?: string | null
  cloneCompletedAt?: string | null
  cloneError?: string | ApiErrorPayload | null
  hasAsrResult?: boolean
  hasCloneResult?: boolean
  elapsedSec?: number | null
  updatedAt?: string | null
  error?: string | ApiErrorPayload | null
  processingModel?: string | null
  asrModel?: string | null
  cloneModel?: string | null
  createdAt: string
}

export interface UploadedFileState extends AudioFileMeta {
  objectUrl?: string
  rawFile?: File
}
