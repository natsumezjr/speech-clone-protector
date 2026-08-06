import axios from 'axios'
import { apiBaseUrl } from '@/config/runtime'
import type { ApiClient } from '@/types/api'
import type { AudioFileMeta } from '@/types/audio'
import type { ApiErrorPayload, AsrEvalRequest, AsrEvalResponse, CloneVoiceRequest, CloneVoiceResult, HistoryTask, ProtectionTaskRequest, PsychoacousticSliceResponse, TaskDetailsResponse, TaskResult, TaskStatusResponse } from '@/types/task'
import { formatStructuredApiError } from '@/utils/apiError'

const http = axios.create({
  baseURL: apiBaseUrl,
})

function filenameFromDisposition(header: string | undefined, fallback: string) {
  if (!header) return fallback
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header)
  return match ? decodeURIComponent(match[1]) : fallback
}

function numberOrNull(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function optionalNumber(value: unknown) {
  return numberOrNull(value)
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {}
}

function stringOr(value: unknown, fallback: string) {
  return typeof value === 'string' && value.length > 0 ? value : fallback
}

function optionalString(value: unknown) {
  return typeof value === 'string' && value.length > 0 ? value : null
}

function firstNumber(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = numberOrNull(record[key])
    if (value !== null) return value
  }
  return null
}

function firstValue(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null && record[key] !== '') return record[key]
  }
  return undefined
}

function firstString(record: Record<string, unknown>, keys: string[]) {
  const value = firstValue(record, keys)
  return typeof value === 'string' && value.length > 0 ? value : null
}

function normalizeRadarPoints(value: unknown): NonNullable<TaskResult['cloneEval']>['cloneRadar'] {
  if (!Array.isArray(value)) return null
  return value
    .map((item, index) => {
      const record = asRecord(item)
      const name = firstString(record, ['name', 'label']) ?? `指标 ${index + 1}`
      const status = firstString(record, ['status']) ?? undefined
      const reason = firstString(record, ['reason'])
      const rawMetricKeys = Array.isArray(record.rawMetricKeys) ? record.rawMetricKeys.filter((key): key is string => typeof key === 'string') : null
      return {
        name,
        value: numberOrNull(record.value),
        status,
        reason,
        formula: firstString(record, ['formula']),
        rawMetricKeys,
      }
    })
    .filter((item) => item.name.length > 0)
}

function normalizeEditCounts(value: unknown): NonNullable<TaskResult['asrEval']>['editCounts'] {
  const data = asRecord(value)
  const referenceLength = numberOrNull(data.referenceLength)
  const substitutions = numberOrNull(data.substitutions)
  const insertions = numberOrNull(data.insertions)
  const deletions = numberOrNull(data.deletions)
  const totalErrors = numberOrNull(data.totalErrors)
  const level = firstString(data, ['level'])
  if (referenceLength === null || substitutions === null || insertions === null || deletions === null || totalErrors === null || !level) return null
  return { level, referenceLength, substitutions, insertions, deletions, totalErrors }
}

function normalizeErrorShares(value: unknown): NonNullable<TaskResult['asrEval']>['errorShares'] {
  const data = asRecord(value)
  const substituteShare = firstNumber(data, ['substituteShare', 'substitutionShare'])
  const insertShare = firstNumber(data, ['insertShare', 'insertionShare'])
  const deleteShare = firstNumber(data, ['deleteShare', 'deletionShare'])
  if (substituteShare === null && insertShare === null && deleteShare === null) return null
  return { substituteShare, insertShare, deleteShare }
}

function parseDateMs(value: unknown) {
  if (typeof value !== 'string' || value.length === 0) return null
  const iso = Date.parse(value)
  if (Number.isFinite(iso)) return iso
  const match = /^(\d{4})\.(\d{1,2})\.(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})$/.exec(value)
  if (!match) return null
  const [, year, month, day, hour, minute, second] = match
  const local = new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second)).getTime()
  return Number.isFinite(local) ? local : null
}

export function normalizeElapsedSec(data: unknown) {
  const record = asRecord(data)
  const direct = firstNumber(record, ['elapsedSec', 'elapsed_sec', 'processingTimeSec', 'processing_time_sec', 'processingTime', 'runtimeSec', 'durationSec', 'timeCostSec'])
  if (direct !== null) return direct
  const start = parseDateMs(firstValue(record, ['startedAt', 'started_at', 'submittedAt', 'submitted_at', 'createdAt', 'created_at']))
  const end = parseDateMs(firstValue(record, ['completedAt', 'completed_at', 'finishedAt', 'finished_at', 'updatedAt', 'updated_at']))
  if (start === null || end === null || end < start) return null
  return (end - start) / 1000
}

function normalizeLossTrendPoint(value: unknown, fallbackStep: number): TaskResult['charts']['optimizationTrend'][number] | null {
  const point = asRecord(value)
  const step = firstNumber(point, ['step', 'epoch', 'iteration', 'iter']) ?? fallbackStep
  const lid = firstNumber(point, ['Lid', 'lId', 'lossIdentity', 'loss_identity', 'Lfeat', 'lFeat', 'Lfea', 'lossFeature', 'loss_timbre', 'L_feature'])
  const normalized = {
    step,
    Lid: lid,
    Lfeat: lid,
    Lsem: firstNumber(point, ['Lsem', 'lSem', 'lossSemantic', 'loss_semantic', 'L_semantic']),
    Lpsy: firstNumber(point, ['Lpsy', 'lPsy', 'lossPsy', 'loss_psy', 'L_psy']),
    L2: firstNumber(point, ['L2', 'l2', 'lossL2', 'loss_l2', 'l2Norm']),
    total: firstNumber(point, ['total', 'totalLoss', 'lossTotal', 'loss_total']),
    snr: firstNumber(point, ['snr', 'SNR']),
    stepElapsedSec: firstNumber(point, ['stepElapsedSec', 'step_elapsed_sec', 'elapsedSec', 'elapsed']),
  }
  const hasLoss = [normalized.Lid, normalized.Lsem, normalized.Lpsy, normalized.L2, normalized.total].some((item) => item !== null)
  return hasLoss ? normalized : null
}

function normalizeLossTrend(value: unknown): TaskResult['charts']['optimizationTrend'] {
  if (!Array.isArray(value)) return []
  const points = value
    .map((point, index) => normalizeLossTrendPoint(point, index))
    .filter((point): point is TaskResult['charts']['optimizationTrend'][number] => point !== null)
  return sampleLossTrend(points)
}

function uniqueLossPoints(points: TaskResult['charts']['optimizationTrend']) {
  const seen = new Set<number>()
  return points.filter((point) => {
    if (seen.has(point.step)) return false
    seen.add(point.step)
    return true
  })
}

function sampleLossTrend(points: TaskResult['charts']['optimizationTrend'], maxPoints = 80): TaskResult['charts']['optimizationTrend'] {
  if (points.length <= maxPoints) return points
  const ordered = [...points].sort((a, b) => a.step - b.step)
  const sampled = Array.from({ length: maxPoints }, (_, index) => {
    const position = Math.round((index / Math.max(1, maxPoints - 1)) * (ordered.length - 1))
    return ordered[position]
  })
  return uniqueLossPoints(sampled)
}

function normalizeLossFinal(value: unknown): NonNullable<TaskResult['generation']>['lossFinal'] {
  const record = asRecord(value)
  const lid = firstNumber(record, ['Lid', 'lId', 'lossIdentity', 'loss_identity', 'Lfeat', 'lFeat', 'Lfea', 'lossFeature', 'loss_timbre', 'L_feature'])
  return {
    Lid: lid,
    Lfeat: lid,
    Lsem: firstNumber(record, ['Lsem', 'lSem', 'lossSemantic', 'loss_semantic', 'L_semantic']),
    Lpsy: firstNumber(record, ['Lpsy', 'lPsy', 'lossPsy', 'loss_psy', 'L_psy']),
    L2: firstNumber(record, ['L2', 'l2', 'lossL2', 'loss_l2', 'l2Norm']),
    total: firstNumber(record, ['total', 'totalLoss', 'lossTotal', 'loss_total']),
    snr: firstNumber(record, ['snr', 'SNR']),
  }
}

function formatApiError(error: unknown): Error {
  if (!axios.isAxiosError(error)) return error instanceof Error ? error : new Error('请求失败。')
  const data = error.response?.data as { error?: ApiErrorPayload; detail?: unknown } | undefined
  const structured = data?.error
  if (structured?.message) {
    return new Error(formatStructuredApiError(structured))
  }
  if (typeof data?.detail === 'string') return new Error(data.detail)
  return new Error(error.message || '请求失败。')
}

http.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(formatApiError(error)),
)

function absoluteUrl(value?: string) {
  if (!value || /^(https?:|blob:|data:)/i.test(value)) return value
  const base = apiBaseUrl.replace(/\/$/, '')
  return `${base}${value.startsWith('/') ? value : `/${value}`}`
}

function normalizeAudio(meta: unknown, fallbackName: string): AudioFileMeta {
  const data = asRecord(meta)
  const filename = firstString(data, ['filename', 'name', 'fileName']) ?? fallbackName
  const rawSrc = firstString(data, ['src', 'url', 'audioUrl', 'downloadUrl']) ?? undefined
  const rawAudioUrl = firstString(data, ['audioUrl', 'url', 'src', 'downloadUrl']) ?? rawSrc
  const rawDownloadUrl = firstString(data, ['downloadUrl', 'url', 'audioUrl', 'src']) ?? undefined
  const rawObjectUrl = typeof data.objectUrl === 'string' ? data.objectUrl : undefined
  return {
    fileId: typeof data.fileId === 'string' ? data.fileId : undefined,
    filename,
    durationSec: firstNumber(data, ['durationSec', 'duration', 'duration_seconds']) ?? undefined,
    duration: firstNumber(data, ['duration', 'durationSec', 'duration_seconds']) ?? undefined,
    sampleRate: firstNumber(data, ['sampleRate', 'sample_rate']) ?? undefined,
    channels: firstNumber(data, ['channels', 'channelCount']) ?? undefined,
    bitDepth: numberOrNull(data.bitDepth) ?? undefined,
    codec: typeof data.codec === 'string' ? data.codec : undefined,
    metadataStatus: data.metadataStatus === 'available' || data.metadataStatus === 'partial' || data.metadataStatus === 'unavailable' ? data.metadataStatus : undefined,
    metadataSource: typeof data.metadataSource === 'string' ? data.metadataSource : undefined,
    metadataReason: typeof data.metadataReason === 'string' ? data.metadataReason : undefined,
    sizeBytes: firstNumber(data, ['sizeBytes', 'size', 'fileSize']) ?? 0,
    format: firstString(data, ['format', 'codec', 'ext']) ?? filename.split('.').pop()?.toUpperCase() ?? 'AUDIO',
    src: absoluteUrl(rawSrc),
    audioUrl: absoluteUrl(rawAudioUrl),
    downloadUrl: absoluteUrl(rawDownloadUrl),
    objectUrl: rawObjectUrl,
    uploadedAt: typeof data.uploadedAt === 'string' ? data.uploadedAt : undefined,
    fingerprint: typeof data.fingerprint === 'string' ? data.fingerprint : undefined,
  }
}

function normalizePerturbation(value: unknown): TaskResult['perturbation'] {
  const record = asRecord(value)
  return {
    l2Norm: firstNumber(record, ['l2Norm', 'l2_norm', 'L2', 'l2']),
    l2Rms: firstNumber(record, ['l2Rms', 'l2_rms']),
    linfNorm: firstNumber(record, ['linfNorm', 'linf_norm', 'lInfNorm']),
    epsilon: firstNumber(record, ['epsilon', 'eps']),
    epsilonNorm: firstString(record, ['epsilonNorm', 'epsilon_norm']) ?? null,
    epsilonUsageRate: firstNumber(record, ['epsilonUsageRate', 'epsilon_usage_rate']),
    snr: firstNumber(record, ['snr', 'SNR']),
    clippingRate: firstNumber(record, ['clippingRate', 'clipping_rate']),
  }
}

function normalizeProtectionQuality(value: unknown): TaskResult['protectionQuality'] {
  const record = asRecord(value)
  return {
    snr: firstNumber(record, ['snr', 'SNR']),
    pesq: firstNumber(record, ['pesq', 'PESQ']),
    stoi: firstNumber(record, ['stoi', 'STOI']),
    mos: firstNumber(record, ['mos', 'MOS']),
    mosLqo: firstNumber(record, ['mosLqo', 'mos_lqo', 'MOSLQO']),
    qualityLevel: firstString(record, ['qualityLevel', 'quality_level']) ?? null,
  }
}

function normalizePsychoacoustic(value: unknown): TaskResult['psychoacoustic'] {
  const record = asRecord(value)
  return {
    lPsy: firstNumber(record, ['lPsy', 'Lpsy', 'lossPsy']),
    overMaskRate: firstNumber(record, ['overMaskRate', 'over_mask_rate', 'psychoacousticViolationRate']),
    frameCount: firstNumber(record, ['frameCount', 'frame_count']),
    sampleRate: firstNumber(record, ['sampleRate', 'sample_rate']),
    hopLength: firstNumber(record, ['hopLength', 'hop_length']),
    nFft: firstNumber(record, ['nFft', 'n_fft', 'nfft']),
    aggregation: firstString(record, ['aggregation']),
    maskingThreshold: Array.isArray(record.maskingThreshold) ? (record.maskingThreshold as NonNullable<TaskResult['psychoacoustic']>['maskingThreshold']) : null,
    perturbationSpectrum: Array.isArray(record.perturbationSpectrum) ? (record.perturbationSpectrum as NonNullable<TaskResult['psychoacoustic']>['perturbationSpectrum']) : null,
  }
}

function normalizeLossWeights(value: unknown): TaskResult['lossWeights'] {
  const record = asRecord(value)
  const lambdaId = firstNumber(record, ['lambdaId', 'lambdaIdentity', 'weightIdentity', 'weight_identity', 'lambdaFeat', 'weightFeature', 'weight_feature'])
  return {
    lambdaId,
    lambdaFeat: lambdaId,
    lambdaSem: firstNumber(record, ['lambdaSem', 'weightSemantic', 'weight_semantic']),
    lambdaPsy: firstNumber(record, ['lambdaPsy', 'weightPsy', 'weight_psy']),
    lambda2: firstNumber(record, ['lambda2', 'weightL2', 'weight_l2']),
  }
}

function hasAnyNumber(record: Record<string, unknown>, keys: string[]) {
  return keys.some((key) => numberOrNull(record[key]) !== null)
}

function normalizeAsrEval(value: unknown): TaskResult['asrEval'] {
  const data = asRecord(value)
  const status = firstString(data, ['status'])
  const referenceText = firstString(data, ['referenceText', 'reference_text'])
  const originalText = firstString(data, ['originalText', 'original_transcript', 'cleanTranscription', 'beforeText'])
  const protectedText = firstString(data, ['protectedText', 'protected_transcript', 'protectedTranscription', 'afterText'])
  const hasMetric = hasAnyNumber(data, ['wer', 'WER', 'cer', 'CER', 'insertRate', 'ir', 'insertionRate', 'deleteRate', 'dr', 'deletionRate', 'substituteRate', 'sr', 'substitutionRate', 'tokenErrorRate', 'token_error_rate', 'tokenChangeRate', 'token_change_rate', 'semanticDrift', 'semantic_drift'])
  if (status && !['computed', 'partial', 'completed', 'success', 'finished'].includes(status) && !originalText && !protectedText && !hasMetric) return null
  if (!status && !originalText && !protectedText && !hasMetric) return null
  return {
    model: firstString(data, ['model', 'asrModel', 'asr_model']) ?? undefined,
    asrModel: firstString(data, ['asrModel', 'asr_model', 'model']),
    language: firstString(data, ['language', 'lang']),
    referenceText,
    originalText,
    protectedText,
    wer: firstNumber(data, ['wer', 'WER']),
    cer: firstNumber(data, ['cer', 'CER']),
    substituteRate: firstNumber(data, ['substituteRate', 'sr', 'substitutionRate']),
    insertRate: firstNumber(data, ['insertRate', 'ir', 'insertionRate']),
    deleteRate: firstNumber(data, ['deleteRate', 'dr', 'deletionRate']),
    editCounts: normalizeEditCounts(data.editCounts),
    errorShares: normalizeErrorShares(data.errorShares),
    metricLevel: firstString(data, ['metricLevel', 'metric_level']),
    tokenErrorRate: firstNumber(data, ['tokenErrorRate', 'token_error_rate']),
    tokenChangeRate: firstNumber(data, ['tokenChangeRate', 'token_change_rate']),
    semanticDrift: firstNumber(data, ['semanticDrift', 'semantic_drift']),
    asrProtectionScore: firstNumber(data, ['asrProtectionScore', 'asr_protection_score']),
    diffOps: Array.isArray(data.diffOps) ? (data.diffOps as NonNullable<TaskResult['asrEval']>['diffOps']) : null,
    trend: Array.isArray(data.trend) ? (data.trend as NonNullable<TaskResult['asrEval']>['trend']) : null,
    createdAt: firstString(data, ['createdAt', 'created_at']),
    status: status ?? undefined,
    error: firstString(data, ['error']),
    reason: firstString(data, ['reason']),
  }
}

function normalizeCloneEval(value: unknown): TaskResult['cloneEval'] {
  const data = asRecord(value)
  const request = asRecord(data.request)
  const originalCloneAudio = data.originalCloneAudio ? normalizeAudio(data.originalCloneAudio, 'original_clone.wav') : null
  const protectedCloneAudio = data.protectedCloneAudio ? normalizeAudio(data.protectedCloneAudio, 'protected_clone.wav') : null
  const hasMetric = hasAnyNumber(data, ['originalSimilarity', 'simBefore', 'similarityBefore', 'protectedSimilarity', 'simAfter', 'similarityAfter', 'embeddingDistanceBefore', 'distanceBefore', 'embeddingDistanceAfter', 'distanceAfter', 'cloneConfidenceBefore', 'confidenceBefore', 'cloneConfidenceAfter', 'confidenceAfter'])
  if (!originalCloneAudio && !protectedCloneAudio && !hasMetric && !data.cloneEval) return null
  return {
    cloneModel: firstString(data, ['cloneModel', 'clone_model']) ?? firstString(request, ['model']),
    speakerEvalModel: firstString(data, ['speakerEvalModel', 'speaker_eval_model']),
    speakerModel: firstString(data, ['speakerModel', 'speaker_model']),
    targetText: firstString(data, ['targetText', 'target_text']) ?? firstString(request, ['text']),
    originalCloneAudio,
    protectedCloneAudio,
    originalSimilarity: firstNumber(data, ['originalSimilarity', 'simBefore', 'similarityBefore']),
    protectedSimilarity: firstNumber(data, ['protectedSimilarity', 'simAfter', 'similarityAfter']),
    similarityDropRate: firstNumber(data, ['similarityDropRate', 'similarity_drop_rate', 'simDropRate']),
    embeddingDistanceBefore: firstNumber(data, ['embeddingDistanceBefore', 'distanceBefore']),
    embeddingDistanceAfter: firstNumber(data, ['embeddingDistanceAfter', 'distanceAfter']),
    embeddingDistanceIncreaseRate: firstNumber(data, ['embeddingDistanceIncreaseRate', 'embedding_distance_increase_rate']),
    cloneConfidenceBefore: firstNumber(data, ['cloneConfidenceBefore', 'confidenceBefore']),
    cloneConfidenceAfter: firstNumber(data, ['cloneConfidenceAfter', 'confidenceAfter']),
    cloneConfidenceDropRate: firstNumber(data, ['cloneConfidenceDropRate', 'confidenceDropRate']),
    cloneRadar: normalizeRadarPoints(data.cloneRadar),
    cloneTrend: null,
    cloneDefenseScore: firstNumber(data, ['cloneDefenseScore', 'clone_defense_score']),
    createdAt: firstString(data, ['createdAt', 'created_at']),
    status: firstString(data, ['status']),
    reason: firstString(data, ['reason', 'error']),
  }
}

function normalizeCloneResult(payload: unknown): CloneVoiceResult {
  const data = asRecord(payload)
  const request = asRecord(data.request)
  const cloneEval = normalizeCloneEval(data)
  return {
    cloneId: String(data.cloneId ?? ''),
    taskId: String(data.taskId ?? ''),
    status: stringOr(data.status, 'partial') as CloneVoiceResult['status'],
    source: typeof data.source === 'string' ? data.source : undefined,
    message: typeof data.message === 'string' ? data.message : undefined,
    request: {
      text: String(request.text ?? ''),
      model: String(request.model ?? 'default'),
      language: typeof request.language === 'string' ? request.language : undefined,
      speed: numberOrNull(request.speed) ?? 1,
      speakerPrompt: typeof request.speakerPrompt === 'string' ? request.speakerPrompt : undefined,
    },
    originalCloneAudio: normalizeAudio(data.originalCloneAudio, 'original_clone.wav'),
    protectedCloneAudio: normalizeAudio(data.protectedCloneAudio, 'protected_clone.wav'),
    cloneEval,
  }
}

function normalizeTaskStatus(payload: unknown): TaskStatusResponse {
  const data = asRecord(payload)
  const cloneTask = asRecord(data.cloneTask)

  return {
    ...(data as unknown as TaskStatusResponse),
    cloneResult:
      data.cloneResult === null || data.cloneResult === undefined
        ? data.cloneResult
        : normalizeCloneResult(data.cloneResult),
    cloneTask:
      data.cloneTask === null || data.cloneTask === undefined
        ? data.cloneTask
        : {
            ...(cloneTask as unknown as NonNullable<TaskStatusResponse['cloneTask']>),
            cloneResult:
              cloneTask.cloneResult === null || cloneTask.cloneResult === undefined
                ? cloneTask.cloneResult
                : normalizeCloneResult(cloneTask.cloneResult),
          },
  }
}

function normalizeTaskResult(payload: unknown): TaskResult {
  const data = asRecord(payload)
  if (data.originalAudio && data.protectedAudio) {
    const originalAudio = asRecord(data.originalAudio)
    const protectedAudio = asRecord(data.protectedAudio)
    const charts = asRecord(data.charts)
    const details = asRecord(data.details)
    const generation = asRecord(data.generation ?? asRecord(details.generation))
    const optimizationTrend = normalizeLossTrend(charts.optimizationTrend ?? asRecord(details.generation).optimizationTrace ?? charts.trend)
    const cloneResults = Array.isArray(data.cloneResults) ? data.cloneResults.map(normalizeCloneResult) : undefined
    const asrEval = normalizeAsrEval(data.asrEval ?? data.asr)
    const latestCloneEval = normalizeCloneEval(data.cloneEval) ?? cloneResults?.at(-1)?.cloneEval ?? null
    return {
      ...(data as unknown as TaskResult),
      originalAudio: normalizeAudio(originalAudio, stringOr(originalAudio.filename, 'original.wav')),
      protectedAudio: normalizeAudio(protectedAudio, stringOr(protectedAudio.filename, 'protected.wav')),
      elapsedSec: normalizeElapsedSec(data),
      perturbation: normalizePerturbation(data.perturbation ?? asRecord(details.perception)),
      protectionQuality: normalizeProtectionQuality(data.protectionQuality ?? data.quality ?? asRecord(details.perception)),
      psychoacoustic: normalizePsychoacoustic(data.psychoacoustic ?? asRecord(details.perception)),
      lossFinal: normalizeLossFinal(data.lossFinal ?? generation.lossFinal),
      lossWeights: normalizeLossWeights(data.lossWeights ?? generation.lossWeights),
      optimizationTrace: optimizationTrend,
      averageStepSec: firstNumber(data, ['averageStepSec', 'average_step_sec']) ?? firstNumber(generation, ['averageStepSec', 'average_step_sec']),
      selectedStep: firstNumber(data, ['selectedStep', 'selected_step']) ?? firstNumber(generation, ['selectedStep', 'selected_step']),
      effectiveConfig: asRecord(data.effectiveConfig ?? data.effective_config ?? generation.effectiveConfig ?? generation.effective_config),
      presetName: stringOr(data.presetName ?? data.preset_name ?? generation.presetName ?? generation.preset_name, '') || null,
      asrEval,
      cloneEval: latestCloneEval,
      speakerFeatureMap: {
        radar: normalizeRadarPoints(asRecord(data.speakerFeatureMap).radar ?? asRecord(details.speakerFeatureMap).radar),
      },
      generation: {
        ...(asRecord(data.generation) as NonNullable<TaskResult['generation']>),
        lossFinal: normalizeLossFinal(generation.lossFinal),
        optimizationTrace: normalizeLossTrend(generation.optimizationTrace),
        steps: numberOrNull(generation.steps),
        maxSteps: firstNumber(generation, ['maxSteps', 'max_steps']),
        selectedStep: firstNumber(generation, ['selectedStep', 'selected_step']),
        snrDb: firstNumber(generation, ['snrDb', 'snr_db', 'snr']),
        presetName: stringOr(generation.presetName ?? generation.preset_name, '') || null,
        effectiveConfig: asRecord(generation.effectiveConfig ?? generation.effective_config),
        realProtect: typeof generation.realProtect === 'boolean' ? generation.realProtect : null,
        source: typeof generation.source === 'string' ? generation.source : undefined,
        status: typeof generation.status === 'string' ? generation.status : undefined,
        mode: typeof generation.mode === 'string' ? generation.mode : undefined,
      },
      charts: {
        ...(charts as TaskResult['charts']),
        psychoacoustic: Array.isArray(charts.psychoacoustic) ? charts.psychoacoustic as TaskResult['charts']['psychoacoustic'] : [],
        trend: Array.isArray(charts.trend) ? charts.trend as TaskResult['charts']['trend'] : [],
        optimizationTrend,
        radarBefore: Array.isArray(charts.radarBefore) ? charts.radarBefore as number[] : undefined,
        radarAfter: Array.isArray(charts.radarAfter) ? charts.radarAfter as number[] : undefined,
        chainRadar: Array.isArray(charts.chainRadar) ? charts.chainRadar as TaskResult['charts']['chainRadar'] : [],
        speakerRadar: normalizeRadarPoints(charts.speakerRadar),
      },
      cloneResults,
    }
  }

  const summary = asRecord(data.summary)
  const primary = asRecord(summary.primaryMetrics)
  const details = asRecord(data.details)
  const audio = asRecord(data.audio)
  const detailAsr = asRecord(details.asr)
  const detailSpeaker = asRecord(details.speaker)
  const detailGeneration = asRecord(details.generation)
  const backend = asRecord(data.backend)
  const charts = asRecord(data.charts)
  const optimizationTrend = normalizeLossTrend(charts.optimizationTrend ?? detailGeneration.optimizationTrace ?? charts.trend)
  const originalRaw = asRecord(audio.original)
  const protectedRaw = asRecord(audio.protected)
  const metricSources = asRecord(summary.metricSources)
  const score = numberOrNull(summary.score)
  const snr = numberOrNull(primary.snr)
  const pesq = numberOrNull(primary.pesq)
  const simAfter = numberOrNull(primary.speakerSimilarity) ?? firstNumber(detailSpeaker, ['simAfter', 'simOriginalProtected'])
  const simBefore = firstNumber(detailSpeaker, ['simBefore'])
  const referenceText = optionalString(detailAsr.referenceText)
  const originalText = optionalString(detailAsr.originalText ?? detailAsr.cleanTranscription)
  const protectedText = optionalString(detailAsr.protectedTranscription)
  const cloneResults = Array.isArray(data.cloneResults) ? data.cloneResults.map(normalizeCloneResult) : undefined
  const lossFinal = normalizeLossFinal(detailGeneration.lossFinal)
  const lossWeights = normalizeLossWeights(detailGeneration.lossWeights)
  const elapsedSec = normalizeElapsedSec(data)
  const asrEval = normalizeAsrEval(detailAsr)
  const asrHasResult = asrEval !== null
  const cloneEval = normalizeCloneEval(data.cloneEval) ?? cloneResults?.at(-1)?.cloneEval ?? null
  const perturbation = normalizePerturbation({
    ...asRecord(details.perception),
    ...asRecord(details.perturbation),
    l2Norm: asRecord(details.perception).l2Norm ?? lossFinal?.L2,
    epsilon: asRecord(asRecord(data.request).optimization).epsilon,
    epsilonNorm: asRecord(asRecord(data.request).optimization).epsilonNorm,
    snr,
  })
  const protectionQuality = normalizeProtectionQuality({
    ...asRecord(details.perception),
    snr,
    pesq,
    mosLqo: asRecord(details.perception).mosLqo,
  })
  const psychoacoustic = normalizePsychoacoustic({
    ...asRecord(details.perception),
    lPsy: lossFinal?.Lpsy,
    overMaskRate: asRecord(details.perception).psychoacousticViolationRate,
  })

  return {
    taskId: String(data.taskId ?? ''),
    status: stringOr(data.status, 'completed') as TaskResult['status'],
    mode: stringOr(data.mode, 'joint') as TaskResult['mode'],
    dataMode: stringOr(data.dataMode, 'backend') as TaskResult['dataMode'],
    verdict: stringOr(summary.verdict, '防护结果已生成'),
    score,
    createdAt: typeof data.createdAt === 'string' ? data.createdAt : undefined,
    submittedAt: typeof data.submittedAt === 'string' ? data.submittedAt : typeof data.createdAt === 'string' ? data.createdAt : undefined,
    completedAt: stringOr(data.completedAt ?? data.createdAt, '-'),
    elapsedSec,
    inputSource: '后端 API',
    language: stringOr(detailAsr.language, '未标注'),
    processingModel: stringOr(detailGeneration.source ?? backend.version, ''),
    optimizationTarget: stringOr(detailGeneration.mode ?? data.mode, 'joint'),
    asrModel: typeof detailAsr.model === 'string' ? detailAsr.model : undefined,
    artifacts: [
      { label: '原始音频', filename: stringOr(originalRaw.filename, 'original.wav'), sizeBytes: numberOrNull(originalRaw.sizeBytes) ?? undefined },
      { label: '保护音频', filename: stringOr(protectedRaw.filename, 'protected.wav'), sizeBytes: numberOrNull(protectedRaw.sizeBytes) ?? undefined },
      { label: '结果 JSON', filename: 'result.json' },
    ],
    originalAudio: normalizeAudio(originalRaw, 'original.wav'),
    protectedAudio: normalizeAudio(protectedRaw, 'protected.wav'),
    perturbation,
    protectionQuality,
    psychoacoustic,
    lossFinal,
    lossWeights,
    optimizationTrace: optimizationTrend,
    averageStepSec: firstNumber(data, ['averageStepSec', 'average_step_sec']) ?? firstNumber(detailGeneration, ['averageStepSec', 'average_step_sec']) ?? null,
    asrEval,
    cloneEval,
    cloneResults,
    speakerFeatureMap: {
      radar: normalizeRadarPoints(asRecord(data.speakerFeatureMap).radar ?? asRecord(details.speakerFeatureMap).radar),
    },
    asr: {
      referenceText: asrHasResult ? referenceText : null,
      originalText: asrHasResult ? originalText : null,
      protectedText: asrHasResult ? protectedText : null,
      wer: asrHasResult ? optionalNumber(primary.wer) ?? optionalNumber(detailAsr.wer) : null,
      cer: asrHasResult ? optionalNumber(primary.cer) ?? optionalNumber(detailAsr.cer) : null,
      tokenErrorRate: asrHasResult ? optionalNumber(detailAsr.tokenErrorRate) ?? optionalNumber(primary.tokenErrorRate) : null,
      semanticDrift: asrHasResult ? optionalNumber(detailAsr.semanticDrift) ?? optionalNumber(primary.semanticDrift) : null,
      insertRate: asrHasResult ? optionalNumber(asRecord(detailAsr.breakdown).insertRate) : null,
      deleteRate: asrHasResult ? optionalNumber(asRecord(detailAsr.breakdown).deleteRate) : null,
      substituteRate: asrHasResult ? optionalNumber(asRecord(detailAsr.breakdown).substituteRate) : null,
      editCounts: asrHasResult ? normalizeEditCounts(detailAsr.editCounts) : null,
      errorShares: asrHasResult ? normalizeErrorShares(detailAsr.errorShares) : null,
      status: typeof detailAsr.status === 'string' ? detailAsr.status : undefined,
    },
    speaker: {
      simBefore,
      simAfter,
      simDropRate: firstNumber(detailSpeaker, ['simDropRate']) ?? optionalNumber(asRecord(details.downstreamTts).simDropRate),
      embeddingDistanceBefore: firstNumber(detailSpeaker, ['embeddingDistanceBefore']),
      embeddingDistanceAfter: firstNumber(detailSpeaker, ['embeddingDistanceAfter', 'embeddingDistance']),
      simOriginalProtected: firstNumber(detailSpeaker, ['simOriginalProtected', 'simAfter']),
      embeddingDistance: firstNumber(detailSpeaker, ['embeddingDistance', 'embeddingDistanceAfter']),
      source: typeof asRecord(metricSources['speaker.*']).source === 'string' ? asRecord(metricSources['speaker.*']).source as string : undefined,
      status: typeof detailSpeaker.status === 'string' ? detailSpeaker.status : undefined,
    },
    quality: {
      snr,
      pesq,
      mosLqo: optionalNumber(asRecord(details.perception).mosLqo),
      l2Norm: optionalNumber(asRecord(details.perception).l2Norm),
      psychoacousticViolationRate: optionalNumber(asRecord(details.perception).psychoacousticViolationRate),
      status: typeof asRecord(details.perception).status === 'string' ? asRecord(details.perception).status as string : undefined,
    },
    metricSources: metricSources as TaskResult['metricSources'],
    selectedStep: firstNumber(data, ['selectedStep', 'selected_step']) ?? firstNumber(detailGeneration, ['selectedStep', 'selected_step']),
    effectiveConfig: asRecord(data.effectiveConfig ?? data.effective_config ?? detailGeneration.effectiveConfig ?? detailGeneration.effective_config),
    presetName: stringOr(data.presetName ?? data.preset_name ?? detailGeneration.presetName ?? detailGeneration.preset_name, '') || null,
    generation: {
      lossFinal,
      optimizationTrace: normalizeLossTrend(detailGeneration.optimizationTrace),
      steps: numberOrNull(detailGeneration.steps),
      maxSteps: firstNumber(detailGeneration, ['maxSteps', 'max_steps']),
      selectedStep: firstNumber(detailGeneration, ['selectedStep', 'selected_step']),
      snrDb: firstNumber(detailGeneration, ['snrDb', 'snr_db', 'snr']),
      presetName: stringOr(detailGeneration.presetName ?? detailGeneration.preset_name, '') || null,
      effectiveConfig: asRecord(detailGeneration.effectiveConfig ?? detailGeneration.effective_config),
      realProtect: typeof detailGeneration.realProtect === 'boolean' ? detailGeneration.realProtect : null,
      source: typeof detailGeneration.source === 'string' ? detailGeneration.source : undefined,
      status: typeof detailGeneration.status === 'string' ? detailGeneration.status : undefined,
      mode: typeof detailGeneration.mode === 'string' ? detailGeneration.mode : undefined,
    },
    raw: data,
    charts: {
      psychoacoustic: Array.isArray(charts.psychoacoustic) ? charts.psychoacoustic as TaskResult['charts']['psychoacoustic'] : [],
      trend: Array.isArray(charts.trend) ? charts.trend as TaskResult['charts']['trend'] : [],
      optimizationTrend,
      radarBefore: Array.isArray(charts.radarBefore) ? charts.radarBefore as number[] : undefined,
      radarAfter: Array.isArray(charts.radarAfter) ? charts.radarAfter as number[] : undefined,
      chainRadar: Array.isArray(charts.chainRadar) ? charts.chainRadar as TaskResult['charts']['chainRadar'] : [],
      speakerRadar: normalizeRadarPoints(charts.speakerRadar),
    },
  }
}

function normalizeHistoryTask(payload: unknown): HistoryTask {
  const data = asRecord(payload)
  const protectionElapsedSec = normalizeElapsedSec({
    ...data,
    elapsedSec: data.protectionElapsedSec ?? data.elapsedSec,
  })
  const asrElapsedSec = normalizeElapsedSec({
    elapsedSec: data.asrElapsedSec,
    startedAt: data.asrStartedAt,
    completedAt: data.asrCompletedAt,
  })
  const cloneElapsedSec = normalizeElapsedSec({
    elapsedSec: data.cloneElapsedSec,
    startedAt: data.cloneStartedAt,
    completedAt: data.cloneCompletedAt,
  })
  return {
    ...(data as unknown as HistoryTask),
    protectionElapsedSec,
    asrElapsedSec,
    cloneElapsedSec,
    elapsedSec: protectionElapsedSec,
  }
}

export const backendClient: ApiClient = {
  async getCapabilities() {
    const response = await http.get('/api/capabilities')
    return response.data
  },
  async uploadFile(file: File): Promise<AudioFileMeta> {
    const form = new FormData()
    form.append('file', file)
    const response = await http.post<AudioFileMeta>('/api/files/upload', form)
    return response.data
  },
  async createProtectionTask(payload: ProtectionTaskRequest) {
    const response = await http.post('/api/tasks/protect', payload)
    return response.data
  },
  async retryProtectionTask(taskId: string) {
    const response = await http.post(`/api/tasks/${taskId}/retry`, undefined, {
      validateStatus: (status) => (status >= 200 && status < 300) || status === 404,
    })
    if (response.status !== 404) return response.data

    const statusResponse = await http.get(`/api/tasks/${taskId}/status`)
    const originalPayload = asRecord(statusResponse.data).payload
    if (typeof originalPayload !== 'object' || originalPayload === null || Array.isArray(originalPayload)) {
      throw new Error('原任务缺少保护参数，无法重试。')
    }
    const fallbackResponse = await http.post('/api/tasks/protect', originalPayload)
    return fallbackResponse.data
  },
  async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    const response = await http.get(`/api/tasks/${taskId}/status`)
    return normalizeTaskStatus(response.data)
  },
  async getTaskResult(taskId: string): Promise<TaskResult> {
    const response = await http.get(`/api/tasks/${taskId}/result`)
    return normalizeTaskResult(response.data)
  },
  async getPsychoacousticSlice(taskId: string, params): Promise<PsychoacousticSliceResponse> {
    const response = await http.get(`/api/tasks/${taskId}/psychoacoustic-slice`, { params })
    const data = asRecord(response.data)
    return {
      ...(data as unknown as PsychoacousticSliceResponse),
      maskingThreshold: Array.isArray(data.maskingThreshold) ? (data.maskingThreshold as PsychoacousticSliceResponse['maskingThreshold']) : [],
      perturbationSpectrum: Array.isArray(data.perturbationSpectrum) ? (data.perturbationSpectrum as PsychoacousticSliceResponse['perturbationSpectrum']) : [],
      charts: asRecord(data.charts) as PsychoacousticSliceResponse['charts'],
    }
  },
  async getTaskDetails(taskId: string): Promise<TaskDetailsResponse> {
    const response = await http.get(`/api/tasks/${taskId}/details`)
    return response.data
  },
  async runAsrEval(taskId: string, payload: AsrEvalRequest): Promise<AsrEvalResponse> {
    const response = await http.post(`/api/tasks/${taskId}/asr-eval`, payload)
    return response.data
  },
  async cloneVoice(taskId: string, payload: CloneVoiceRequest): Promise<CloneVoiceResult> {
    const response = await http.post(`/api/tasks/${taskId}/clone-voice`, payload)
    return normalizeCloneResult(response.data)
  },
  async listTasks(): Promise<HistoryTask[]> {
    const response = await http.get('/api/tasks')
    return Array.isArray(response.data) ? response.data.map(normalizeHistoryTask) : []
  },
  async deleteTask(taskId: string): Promise<void> {
    await http.delete(`/api/tasks/${taskId}`)
  },
  async downloadProtectedAudio(taskId: string) {
    const response = await http.get(`/api/tasks/${taskId}/download?type=protected_audio`, {
      responseType: 'blob',
    })
    return {
      blob: response.data,
      filename: filenameFromDisposition(response.headers['content-disposition'], 'protected_voice.wav'),
    }
  },
  async exportReport(taskId: string): Promise<Blob> {
    const response = await http.get(`/api/tasks/${taskId}/download?type=report_pdf`, { responseType: 'blob' })
    return response.data
  },
  async exportCsv(taskId: string): Promise<Blob> {
    const response = await http.get(`/api/tasks/${taskId}/export/csv`, { responseType: 'blob' })
    return response.data
  },
  async downloadEvidenceZip(taskId: string): Promise<Blob> {
    const response = await http.get(`/api/tasks/${taskId}/download?type=evidence_zip`, { responseType: 'blob' })
    return response.data
  },
}
