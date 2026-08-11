import type { AudioFileMeta } from './audio'

export type DataMode = 'backend'

export type TaskStatus = 'queued' | 'running' | 'completed' | 'success' | 'partial_failed' | 'failed' | 'error' | 'cancelled'
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
  request?: AsrEvalRequest
  createdAt?: string | null
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
  asrRequest?: AsrEvalRequest | null
  cloneRequest?: CloneVoiceRequest | null
  asrResult?: AsrEvalResponse | null
  cloneResult?: CloneVoiceResult | null
}

export type EvaluationBatchType = 'asr' | 'clone'
export type EvaluationBatchStatus = TaskStatus

export interface EvaluationBatchItem {
  batchItemId: string
  model: string
  modelName?: string | null
  modelType?: string | null
  annotationSource?: 'manual' | 'asr' | null
  status: TaskStatus | string
  progress?: number | null
  message?: string | null
  elapsedSec?: number | null
  error?: string | ApiErrorPayload | null
  asrSubId?: string | null
  cloneSubId?: string | null
  resultRef?: string | null
}

export interface EvaluationBatch {
  batchId: string
  type: EvaluationBatchType
  taskId: string
  label: string
  status: EvaluationBatchStatus | string
  progress: number
  elapsedSec?: number | null
  completedCount: number
  failedCount: number
  totalCount: number
  createdAt?: string | null
  updatedAt?: string | null
  items: EvaluationBatchItem[]
}

export interface CreateEvaluationBatchRequest {
  batchId: string
  type: EvaluationBatchType
  items: Array<{
    batchItemId: string
    model: string
    modelName?: string
    modelType?: string
    annotationSource?: 'manual' | 'asr'
  }>
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
  asrTasks?: SubtaskStatusSnapshot[]
  cloneTasks?: SubtaskStatusSnapshot[]
  asrBatches?: EvaluationBatch[]
  cloneBatches?: EvaluationBatch[]
}

export interface SharedSemanticMetrics {
  status?: string | null
  tokenChangeRate?: number | null
  tokenErrorRate?: number | null
  tokenChangeCount?: number | null
  tokenTotal?: number | null
  semanticDrift?: number | null
  tokenScore?: number | null
  driftScore?: number | null
  protectionSemanticScore?: number | null
  scoreStatus?: string | null
  scoreReason?: string | null
  encoderDistances?: Array<Record<string, unknown>> | null
  reason?: string | null
  error?: string | null
}

export interface SpeakerMetrics {
  simBefore: number | null
  simAfter: number | null
  simDropRate: number | null
  embeddingDistanceBefore: number | null
  embeddingDistanceAfter: number | null
  simOriginalProtected?: number | null
  embeddingDistance?: number | null
  directDistance?: number | null
  directIdentityScore?: number | null
  scoreStatus?: string | null
  scoreReason?: string | null
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
  epsilonUsageRateRaw?: number | null
  epsilonToleranceRate?: number | null
  epsilonExceeded?: boolean | null
  snr?: number | null
  clippingRate?: number | null
}

export interface ProtectionQuality {
  snr?: number | null
  pesq?: number | null
  stoi?: number | null
  mos?: number | null
  mosLqo?: number | null
  dnsMos?: number | null
  snrScore?: number | null
  stoiScore?: number | null
  pesqScore?: number | null
  mosScore?: number | null
  dnsMosScore?: number | null
  qualityScore?: number | null
  dnsMosStatus?: string | null
  dnsMosReason?: string | null
  scoreStatus?: string | null
  scoreReason?: string | null
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
  directSimilarity?: number | null
  originalSimilarity?: number | null
  protectedSimilarity?: number | null
  similarityDropRate?: number | null
  embeddingDistanceBefore?: number | null
  embeddingDistanceAfter?: number | null
  embeddingDistanceDelta?: number | null
  embeddingDistanceIncreaseRate?: number | null
  cloneConfidenceBefore?: number | null
  cloneConfidenceAfter?: number | null
  cloneConfidenceDropRate?: number | null
  cloneRadar?: RadarPoint[] | null
  cloneTrend?: Array<Record<string, number>> | null
  cloneDefenseScore?: number | null
  cloneIdentityScore?: number | null
  identityBaselineWeight?: number | null
  cloneIdentityStatus?: string | null
  cloneIdentityReason?: string | null
  cleanCloneTranscription?: string | null
  protectedCloneTranscription?: string | null
  cloneAsrModel?: string | null
  cloneAsrStatus?: string | null
  cloneAsrReason?: string | null
  cleanCloneTextAccuracy?: number | null
  cleanCloneTextError?: number | null
  protectedCloneTextAccuracy?: number | null
  protectedCloneTextError?: number | null
  cloneTextChangeAccuracy?: number | null
  cloneTextChangeRate?: number | null
  semanticBaselineWeight?: number | null
  cloneTokenChangeRate?: number | null
  cloneSemanticDrift?: number | null
  cloneTokenScore?: number | null
  cloneDriftScore?: number | null
  cloneSemanticScore?: number | null
  cloneSemanticStatus?: string | null
  cloneSemanticReason?: string | null
  cleanCloneQualityMos?: number | null
  protectedCloneQualityMos?: number | null
  cloneQualityDropRate?: number | null
  cloneQualityRawScore?: number | null
  cloneQualityRelevance?: number | null
  cloneQualityScore?: number | null
  qualityBaselineWeight?: number | null
  cloneQualityModel?: string | null
  cloneQualityModelPath?: string | null
  cloneQualityStatus?: string | null
  cloneQualityReason?: string | null
  createdAt?: string | null
  status?: string | null
  reason?: string | null
  metricSources?: Record<string, MetricSource>
}

export interface MetricSource {
  source?: string
  status?: string
  reason?: string
  formula?: string
  metric?: string
}

export type ProtectionEvaluationDimensionKey =
  | 'protectionQuality'
  | 'cloneQuality'
  | 'protectionSemantic'
  | 'cloneSemantic'
  | 'directIdentity'
  | 'cloneIdentity'

export interface ProtectionEvaluationDimension {
  key: ProtectionEvaluationDimensionKey
  label: string
  score: number | null
  status: 'available' | 'unavailable' | 'pending' | 'error' | string
  reason?: string | null
  weight: number
}

export interface ProtectionEvaluationRecommendation {
  key: string
  message: string
  parameters: string[]
}

export interface ProtectionEvaluationCalibration {
  tokenChangeRate90?: number | null
  semanticDrift90?: number | null
  directDistance90?: number | null
  cloneTokenChangeRate90?: number | null
  cloneSemanticDrift90?: number | null
  cloneQualityDropRate90?: number | null
}

export interface ProtectionEvaluation {
  status: 'complete' | 'incomplete' | string
  overallScore: number | null
  level: '优秀' | '中等' | '较差' | string | null
  verdict: string
  dimensions: ProtectionEvaluationDimension[]
  missingDimensions: string[]
  recommendations: ProtectionEvaluationRecommendation[]
  calibration?: ProtectionEvaluationCalibration | null
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
  name?: string
  value: string
  backendValue?: string
  branch?: string
  backend?: string
  defaultPath?: string
  type?: string[]
  information?: string
  status?: 'available' | 'unavailable' | 'download_required' | string
  reason?: string | null
  languages?: string[]
  requiresReferenceText?: boolean
  promptRequired?: boolean
  annotationSources?: Array<'manual' | 'asr'>
  fineTuneMode?: 'live_fine_tune' | string
}

export interface RuntimeModelType {
  value: string
  name: string
  information: string
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
  modelTypes?: Record<string, RuntimeModelType[]>
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

export interface RuntimeConcurrency {
  protect: number
  asr: number
  clone: number
  asrCloneShared?: number
  protectSharesWorkerGpu?: boolean
  total: number
  unit?: string
  definition?: string
  cloneBackends?: Record<string, number>
  cloneGpuSlots?: {
    limitPerGpu?: number
    keys?: Record<string, string[]>
    asr?: string[]
  }
}

export interface RuntimePerformance {
  averageStepSec?: number | null
  sourceTaskId?: string | null
  source?: string
}

export interface CapabilitiesResponse {
  ok: boolean
  modelTypes?: Record<string, RuntimeModelType[]>
  device?: string
  chains?: Record<string, CapabilityChain>
  config?: ProtectionRuntimeConfig
  defaults?: ProtectionRuntimeConfig['defaults']
  ranges?: ProtectionRuntimeConfig['ranges']
  models?: ProtectionRuntimeConfig['models']
  constraints?: ProtectionRuntimeConfig['constraints']
  runtimeConcurrency?: RuntimeConcurrency
  runtimePerformance?: RuntimePerformance
  protectQueue?: {
    maxConcurrency?: number
    activeCount?: number
    queuedCount?: number
  }
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
  protectionEvaluation?: ProtectionEvaluation | null
  psychoacoustic?: PsychoacousticMetrics | null
  lossFinal?: LossFinal | null
  lossWeights?: LossWeights | null
  optimizationTrace?: LossTrendPoint[] | null
  averageStepSec?: number | null
  selectedStep?: number | null
  effectiveConfig?: Record<string, unknown> | null
  presetName?: string | null
  asrEval?: AsrEval | null
  semanticEval?: SharedSemanticMetrics | null
  asrResults?: AsrEvalResponse[]
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
  originalSpeakerPrompt?: string
  protectedSpeakerPrompt?: string
  annotationSource?: 'manual' | 'asr'
  annotationAsrSubId?: string
  annotationAsrModel?: string
  annotationCreatedAt?: string
  batchId?: string
  batchItemId?: string
}

export interface FineTuneConditionEvidence {
  textSec?: number | null
  hubertSec?: number | null
  semanticSec?: number | null
  s1TrainSec?: number | null
  s2TrainSec?: number | null
  inferenceWallSec?: number | null
  totalWallSec?: number | null
  sourceDurationSec?: number | null
  trainingDurationSec?: number | null
  gptCheckpoint?: string | null
  sovitsCheckpoint?: string | null
  referencePath?: string | null
  trainingAudioPath?: string | null
  outputPath?: string | null
}

export interface FineTuneEvidence {
  model?: string | null
  mode?: string | null
  workDir?: string | null
  pairWallSec?: number | null
  original?: FineTuneConditionEvidence | null
  protected?: FineTuneConditionEvidence | null
}

export interface AsrEvalRequest {
  model: string
  language?: string
  referenceText?: string
  batchId?: string
  batchItemId?: string
}

export interface CloneVoiceResult {
  cloneId: string
  cloneSubId?: string
  taskId: string
  status: 'queued' | 'running' | 'completed' | 'success' | 'partial' | 'failed' | 'error'
  source?: string
  message?: string
  request: CloneVoiceRequest
  originalCloneAudio: AudioFileMeta
  protectedCloneAudio: AudioFileMeta
  cloneEval?: CloneEval | null
  fineTune?: FineTuneEvidence | null
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
  protectionCompletedAt?: string | null
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
  asrTaskCount?: number
  cloneTaskCount?: number
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
