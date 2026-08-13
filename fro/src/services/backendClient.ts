import axios from 'axios'
import { apiBaseUrl } from '@/config/runtime'
import type { ApiClient, EvaluationDeleteResult } from '@/types/api'
import type { AudioFileMeta } from '@/types/audio'
import type { ApiErrorPayload, AsrEvalRequest, AsrEvalResponse, CapabilitiesResponse, CloneVoiceRequest, CloneVoiceResult, CreateEvaluationBatchRequest, EvaluationBatch, HistoryTask, MetricSource, ProtectionEvaluation, ProtectionEvaluationDimension, ProtectionEvaluationDimensionKey, ProtectionTaskRequest, PsychoacousticSliceResponse, TaskDetailsResponse, TaskResult, TaskStatusResponse } from '@/types/task'
import { formatStructuredApiError } from '@/utils/apiError'
import { layeredMetricNumber } from '@/utils/metricNormalization'
import { filenameFromContentDisposition } from '@/utils/contentDisposition'

const http = axios.create({
  baseURL: apiBaseUrl,
})

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
  const filename = firstString(data, ['displayFilename', 'filename', 'name', 'fileName']) ?? fallbackName
  const rawSrc = firstString(data, ['src', 'url', 'audioUrl', 'downloadUrl']) ?? undefined
  const rawAudioUrl = firstString(data, ['audioUrl', 'url', 'src', 'downloadUrl']) ?? rawSrc
  const rawDownloadUrl = firstString(data, ['downloadUrl', 'url', 'audioUrl', 'src']) ?? undefined
  const rawObjectUrl = typeof data.objectUrl === 'string' ? data.objectUrl : undefined
  return {
    fileId: typeof data.fileId === 'string' ? data.fileId : undefined,
    filename,
    displayFilename: firstString(data, ['displayFilename']) ?? filename,
    storedFilename: firstString(data, ['storedFilename']) ?? undefined,
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
    epsilonUsageRateRaw: firstNumber(record, ['epsilonUsageRateRaw', 'epsilon_usage_rate_raw']),
    epsilonToleranceRate: firstNumber(record, ['epsilonToleranceRate', 'epsilon_tolerance_rate']),
    epsilonExceeded: typeof record.epsilonExceeded === 'boolean'
      ? record.epsilonExceeded
      : typeof record.epsilon_exceeded === 'boolean'
        ? record.epsilon_exceeded
        : null,
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
    dnsMos: firstNumber(record, ['dnsMos', 'dns_mos', 'dnsmos']),
    snrScore: firstNumber(record, ['snrScore', 'snr_score']),
    stoiScore: firstNumber(record, ['stoiScore', 'stoi_score']),
    pesqScore: firstNumber(record, ['pesqScore', 'pesq_score']),
    mosScore: firstNumber(record, ['mosScore', 'mos_score']),
    dnsMosScore: firstNumber(record, ['dnsMosScore', 'dns_mos_score']),
    qualityScore: firstNumber(record, ['qualityScore', 'quality_score']),
    dnsMosStatus: firstString(record, ['dnsMosStatus', 'dns_mos_status']),
    dnsMosReason: firstString(record, ['dnsMosReason', 'dns_mos_reason']),
    scoreStatus: firstString(record, ['scoreStatus', 'score_status']),
    scoreReason: firstString(record, ['scoreReason', 'score_reason']),
    qualityLevel: firstString(record, ['qualityLevel', 'quality_level']) ?? null,
  }
}

const protectionDimensionKeys = new Set<ProtectionEvaluationDimensionKey>([
  'protectionQuality',
  'cloneQuality',
  'protectionSemantic',
  'cloneSemantic',
  'directIdentity',
  'cloneIdentity',
])

function normalizeProtectionDimension(value: unknown): ProtectionEvaluationDimension | null {
  const record = asRecord(value)
  const key = firstString(record, ['key'])
  if (!key || !protectionDimensionKeys.has(key as ProtectionEvaluationDimensionKey)) return null
  return {
    key: key as ProtectionEvaluationDimensionKey,
    label: firstString(record, ['label', 'name']) ?? key,
    score: firstNumber(record, ['score', 'value']),
    status: firstString(record, ['status']) ?? 'unavailable',
    reason: firstString(record, ['reason']),
    weight: firstNumber(record, ['weight']) ?? 0,
  }
}

function normalizeProtectionEvaluation(value: unknown): ProtectionEvaluation | null {
  const record = asRecord(value)
  if (Object.keys(record).length === 0) return null
  const dimensions = Array.isArray(record.dimensions)
    ? record.dimensions.map(normalizeProtectionDimension).filter((item): item is ProtectionEvaluationDimension => item !== null)
    : []
  const recommendations = Array.isArray(record.recommendations)
    ? record.recommendations.map((item) => {
        const recommendation = asRecord(item)
        const message = firstString(recommendation, ['message'])
        if (!message) return null
        return {
          key: firstString(recommendation, ['key']) ?? message,
          message,
          parameters: Array.isArray(recommendation.parameters)
            ? recommendation.parameters.filter((parameter): parameter is string => typeof parameter === 'string')
            : [],
        }
      }).filter((item): item is NonNullable<ProtectionEvaluation['recommendations']>[number] => item !== null)
    : []
  const calibrationRecord = asRecord(record.calibration)
  const calibration = Object.keys(calibrationRecord).length
    ? {
        tokenChangeRate90: firstNumber(calibrationRecord, ['tokenChangeRate90', 'token_change_rate_90']),
        semanticDrift90: firstNumber(calibrationRecord, ['semanticDrift90', 'semantic_drift_90']),
        directDistance90: firstNumber(calibrationRecord, ['directDistance90', 'direct_distance_90']),
        cloneTokenChangeRate90: firstNumber(calibrationRecord, ['cloneTokenChangeRate90', 'clone_token_change_rate_90']),
        cloneSemanticDrift90: firstNumber(calibrationRecord, ['cloneSemanticDrift90', 'clone_semantic_drift_90']),
        cloneQualityDropRate90: firstNumber(calibrationRecord, ['cloneQualityDropRate90', 'clone_quality_drop_rate_90']),
        cloneQualityWeightedDrop90: firstNumber(calibrationRecord, ['cloneQualityWeightedDrop90', 'clone_quality_weighted_drop_90']),
      }
    : null
  const overallScore = firstNumber(record, ['overallScore', 'overall_score'])
  const status = firstString(record, ['status']) ?? (overallScore === null ? 'incomplete' : 'complete')
  return {
    status,
    overallScore,
    level: firstString(record, ['level']),
    verdict: firstString(record, ['verdict']) ?? (status === 'complete' ? '' : '待完整评估'),
    dimensions,
    missingDimensions: Array.isArray(record.missingDimensions)
      ? record.missingDimensions.filter((item): item is string => typeof item === 'string')
      : [],
    recommendations,
    calibration,
  }
}

function canonicalTaskScore(payloadScore: unknown, evaluation: ProtectionEvaluation | null) {
  return numberOrNull(evaluation?.overallScore) ?? numberOrNull(payloadScore)
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

function normalizeSemanticEval(value: unknown): TaskResult['semanticEval'] {
  const data = asRecord(value)
  if (Object.keys(data).length === 0) return null
  return {
    status: firstString(data, ['status']),
    tokenChangeRate: firstNumber(data, ['tokenChangeRate', 'token_change_rate']),
    tokenErrorRate: firstNumber(data, ['tokenErrorRate', 'token_error_rate']),
    tokenChangeCount: firstNumber(data, ['tokenChangeCount', 'token_change_count']),
    tokenTotal: firstNumber(data, ['tokenTotal', 'token_total']),
    semanticDrift: firstNumber(data, ['semanticDrift', 'semantic_drift']),
    tokenScore: firstNumber(data, ['tokenScore', 'token_score']),
    driftScore: firstNumber(data, ['driftScore', 'drift_score']),
    protectionSemanticScore: firstNumber(data, ['protectionSemanticScore', 'protection_semantic_score']),
    scoreStatus: firstString(data, ['scoreStatus', 'score_status']),
    scoreReason: firstString(data, ['scoreReason', 'score_reason']),
    encoderDistances: Array.isArray(data.encoderDistances) ? data.encoderDistances as Array<Record<string, unknown>> : null,
    reason: firstString(data, ['reason']),
    error: firstString(data, ['error']),
  }
}

function normalizeCloneEval(value: unknown): TaskResult['cloneEval'] {
  const data = asRecord(value)
  const nestedEval = asRecord(data.cloneEval)
  const metricLayers = [nestedEval, data]
  const metricNumber = (keys: string[]) => layeredMetricNumber(metricLayers, keys)
  const metricString = (keys: string[]) => {
    for (const layer of metricLayers) {
      const result = firstString(layer, keys)
      if (result !== null) return result
    }
    return null
  }
  const metricValue = (keys: string[]) => {
    for (const layer of metricLayers) {
      const result = firstValue(layer, keys)
      if (result !== undefined) return result
    }
    return undefined
  }
  const rawMetricSources = asRecord(metricValue(['metricSources', '_metricSources']))
  const metricSources = Object.keys(rawMetricSources).length > 0
    ? rawMetricSources as Record<string, MetricSource>
    : undefined
  const metricSourceReason = (...keys: string[]) => {
    for (const key of keys) {
      const reason = metricSources?.[key]?.reason
      if (typeof reason === 'string' && reason.length > 0) return reason
    }
    return null
  }
  const request = asRecord(data.request)
  const originalCloneAudioValue = metricValue(['originalCloneAudio'])
  const protectedCloneAudioValue = metricValue(['protectedCloneAudio'])
  const originalCloneAudio = originalCloneAudioValue ? normalizeAudio(originalCloneAudioValue, 'original_clone.wav') : null
  const protectedCloneAudio = protectedCloneAudioValue ? normalizeAudio(protectedCloneAudioValue, 'protected_clone.wav') : null
  const metricKeys = ['directSimilarity', 'direct_similarity', 'originalSimilarity', 'simBefore', 'similarityBefore', 'protectedSimilarity', 'simAfter', 'similarityAfter', 'embeddingDistanceBefore', 'distanceBefore', 'embeddingDistanceAfter', 'distanceAfter', 'cloneConfidenceBefore', 'confidenceBefore', 'cloneConfidenceAfter', 'confidenceAfter', 'cloneIdentityScore', 'cloneSemanticScore', 'cloneQualityScore']
  const hasMetric = metricLayers.some((layer) => hasAnyNumber(layer, metricKeys))
  if (!originalCloneAudio && !protectedCloneAudio && !hasMetric && !data.cloneEval) return null
  return {
    cloneModel: metricString(['cloneModel', 'clone_model']) ?? firstString(request, ['model']),
    speakerEvalModel: metricString(['speakerEvalModel', 'speaker_eval_model']),
    speakerModel: metricString(['speakerModel', 'speaker_model']),
    targetText: metricString(['targetText', 'target_text']) ?? firstString(request, ['text']),
    originalCloneAudio,
    protectedCloneAudio,
    directSimilarity: metricNumber(['directSimilarity', 'direct_similarity']),
    originalSimilarity: metricNumber(['originalSimilarity', 'simBefore', 'similarityBefore']),
    protectedSimilarity: metricNumber(['protectedSimilarity', 'simAfter', 'similarityAfter']),
    similarityDropRate: metricNumber(['similarityDropRate', 'similarity_drop_rate', 'simDropRate']),
    embeddingDistanceBefore: metricNumber(['embeddingDistanceBefore', 'distanceBefore']),
    embeddingDistanceAfter: metricNumber(['embeddingDistanceAfter', 'distanceAfter']),
    embeddingDistanceDelta: metricNumber(['embeddingDistanceDelta', 'embedding_distance_delta']),
    embeddingDistanceIncreaseRate: metricNumber(['embeddingDistanceIncreaseRate', 'embedding_distance_increase_rate']),
    cloneConfidenceBefore: metricNumber(['cloneConfidenceBefore', 'confidenceBefore']),
    cloneConfidenceAfter: metricNumber(['cloneConfidenceAfter', 'confidenceAfter']),
    cloneConfidenceDropRate: metricNumber(['cloneConfidenceDropRate', 'confidenceDropRate']),
    cloneRadar: normalizeRadarPoints(metricValue(['cloneRadar'])),
    cloneTrend: Array.isArray(metricValue(['cloneTrend'])) ? metricValue(['cloneTrend']) as Array<Record<string, number>> : null,
    cloneDefenseScore: metricNumber(['cloneDefenseScore', 'clone_defense_score']),
    cloneIdentityScore: metricNumber(['cloneIdentityScore', 'clone_identity_score']),
    identityBaselineWeight: metricNumber(['identityBaselineWeight', 'identity_baseline_weight']),
    cloneIdentityStatus: metricString(['cloneIdentityStatus', 'clone_identity_status']),
    cloneIdentityReason: metricString(['cloneIdentityReason', 'clone_identity_reason']) ?? metricSourceReason('cloneEval.cloneIdentityScore', 'cloneEval.*'),
    cleanCloneTranscription: metricString(['cleanCloneTranscription', 'clean_clone_transcription']),
    protectedCloneTranscription: metricString(['protectedCloneTranscription', 'protected_clone_transcription']),
    cloneAsrModel: metricString(['cloneAsrModel', 'clone_asr_model']),
    cloneAsrStatus: metricString(['cloneAsrStatus', 'clone_asr_status']),
    cloneAsrReason: metricString(['cloneAsrReason', 'clone_asr_reason']) ?? metricSourceReason('cloneEval.cloneAsr'),
    cleanCloneTextAccuracy: metricNumber(['cleanCloneTextAccuracy', 'clean_clone_text_accuracy']),
    cleanCloneTextError: metricNumber(['cleanCloneTextError', 'clean_clone_text_error']),
    protectedCloneTextAccuracy: metricNumber(['protectedCloneTextAccuracy', 'protected_clone_text_accuracy']),
    protectedCloneTextError: metricNumber(['protectedCloneTextError', 'protected_clone_text_error']),
    cloneTextChangeAccuracy: metricNumber(['cloneTextChangeAccuracy', 'clone_text_change_accuracy']),
    cloneTextChangeRate: metricNumber(['cloneTextChangeRate', 'clone_text_change_rate']),
    semanticBaselineWeight: metricNumber(['semanticBaselineWeight', 'semantic_baseline_weight']),
    cloneTokenChangeRate: metricNumber(['cloneTokenChangeRate', 'clone_token_change_rate']),
    cloneSemanticDrift: metricNumber(['cloneSemanticDrift', 'clone_semantic_drift']),
    cloneTokenScore: metricNumber(['cloneTokenScore', 'clone_token_score']),
    cloneDriftScore: metricNumber(['cloneDriftScore', 'clone_drift_score']),
    cloneSemanticScore: metricNumber(['cloneSemanticScore', 'clone_semantic_score']),
    cloneSemanticStatus: metricString(['cloneSemanticStatus', 'clone_semantic_status']),
    cloneSemanticReason: metricString(['cloneSemanticReason', 'clone_semantic_reason']) ?? metricSourceReason('cloneEval.cloneSemanticScore'),
    cleanCloneQualityMos: metricNumber(['cleanCloneQualityMos', 'clean_clone_quality_mos']),
    protectedCloneQualityMos: metricNumber(['protectedCloneQualityMos', 'protected_clone_quality_mos']),
    clonePairPesq: metricNumber(['clonePairPesq', 'clone_pair_pesq']),
    clonePairStoi: metricNumber(['clonePairStoi', 'clone_pair_stoi']),
    cloneQualityBefore: metricNumber(['cloneQualityBefore', 'clone_quality_before']),
    cloneQualityAfter: metricNumber(['cloneQualityAfter', 'clone_quality_after']),
    cloneQualityDropRate: metricNumber(['cloneQualityDropRate', 'clone_quality_drop_rate']),
    clonePesqDegradationScore: metricNumber(['clonePesqDegradationScore', 'clone_pesq_degradation_score']),
    cloneStoiDegradationScore: metricNumber(['cloneStoiDegradationScore', 'clone_stoi_degradation_score']),
    cloneDnsMosDegradationScore: metricNumber(['cloneDnsMosDegradationScore', 'clone_dnsmos_degradation_score']),
    cloneQualityComponents: (() => {
      const components = asRecord(metricValue(['cloneQualityComponents', 'clone_quality_components']))
      if (!Object.keys(components).length) return null
      const component = (key: 'pesq' | 'stoi' | 'dnsmos') => {
        const value = asRecord(components[key])
        if (!Object.keys(value).length) return null
        return {
          before: firstNumber(value, ['before']),
          after: firstNumber(value, ['after']),
          weight: firstNumber(value, ['weight']),
        }
      }
      return {
        pesq: component('pesq'),
        stoi: component('stoi'),
        dnsmos: component('dnsmos'),
      }
    })(),
    cloneQualityRawScore: metricNumber(['cloneQualityRawScore', 'clone_quality_raw_score']),
    cloneQualityRelevance: metricNumber(['cloneQualityRelevance', 'clone_quality_relevance']),
    cloneQualityScore: metricNumber(['cloneQualityScore', 'clone_quality_score']),
    qualityBaselineWeight: metricNumber(['qualityBaselineWeight', 'quality_baseline_weight']),
    cloneQualityModel: metricString(['cloneQualityModel', 'clone_quality_model']),
    cloneQualityModelPath: metricString(['cloneQualityModelPath', 'clone_quality_model_path']),
    cloneQualityStatus: metricString(['cloneQualityStatus', 'clone_quality_status']),
    cloneQualityReason: metricString(['cloneQualityReason', 'clone_quality_reason']) ?? metricSourceReason('cloneEval.cloneQualityScore'),
    createdAt: metricString(['createdAt', 'created_at']),
    status: metricString(['status']),
    reason: metricString(['reason', 'error']),
    metricSources,
  }
}

function normalizeFineTuneCondition(value: unknown): NonNullable<CloneVoiceResult['fineTune']>['original'] {
  const data = asRecord(value)
  if (Object.keys(data).length === 0) return null
  return {
    textSec: firstNumber(data, ['textSec']),
    hubertSec: firstNumber(data, ['hubertSec']),
    semanticSec: firstNumber(data, ['semanticSec']),
    s1TrainSec: firstNumber(data, ['s1TrainSec']),
    s2TrainSec: firstNumber(data, ['s2TrainSec']),
    inferenceWallSec: firstNumber(data, ['inferenceWallSec']),
    totalWallSec: firstNumber(data, ['totalWallSec']),
    sourceDurationSec: firstNumber(data, ['sourceDurationSec']),
    trainingDurationSec: firstNumber(data, ['trainingDurationSec']),
    gptCheckpoint: firstString(data, ['gptCheckpoint']),
    sovitsCheckpoint: firstString(data, ['sovitsCheckpoint']),
    referencePath: firstString(data, ['referencePath']),
    trainingAudioPath: firstString(data, ['trainingAudioPath']),
    outputPath: firstString(data, ['outputPath']),
  }
}

function normalizeFineTuneEvidence(value: unknown): CloneVoiceResult['fineTune'] {
  const data = asRecord(value)
  if (Object.keys(data).length === 0) return null
  return {
    model: firstString(data, ['model']),
    mode: firstString(data, ['mode']),
    workDir: firstString(data, ['workDir']),
    pairWallSec: firstNumber(data, ['pairWallSec']),
    original: normalizeFineTuneCondition(data.original),
    protected: normalizeFineTuneCondition(data.protected),
  }
}

function normalizeCloneResult(payload: unknown): CloneVoiceResult {
  const data = asRecord(payload)
  const request = asRecord(data.request)
  const cloneEval = normalizeCloneEval(data)
  return {
    cloneId: String(data.cloneId ?? ''),
    cloneSubId: typeof data.cloneSubId === 'string' ? data.cloneSubId : undefined,
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
      originalSpeakerPrompt: typeof request.originalSpeakerPrompt === 'string' ? request.originalSpeakerPrompt : undefined,
      protectedSpeakerPrompt: typeof request.protectedSpeakerPrompt === 'string' ? request.protectedSpeakerPrompt : undefined,
      annotationSource: request.annotationSource === 'asr' ? 'asr' : request.annotationSource === 'manual' ? 'manual' : undefined,
      annotationAsrSubId: typeof request.annotationAsrSubId === 'string' ? request.annotationAsrSubId : undefined,
      annotationAsrModel: typeof request.annotationAsrModel === 'string' ? request.annotationAsrModel : undefined,
      annotationCreatedAt: typeof request.annotationCreatedAt === 'string' ? request.annotationCreatedAt : undefined,
      batchId: typeof request.batchId === 'string' ? request.batchId : undefined,
      batchItemId: typeof request.batchItemId === 'string' ? request.batchItemId : undefined,
    },
    originalCloneAudio: normalizeAudio(data.originalCloneAudio, 'original_clone.wav'),
    protectedCloneAudio: normalizeAudio(data.protectedCloneAudio, 'protected_clone.wav'),
    cloneEval,
    fineTune: normalizeFineTuneEvidence(data.fineTune),
  }
}

function normalizeAsrResult(payload: unknown): AsrEvalResponse {
  const data = asRecord(payload)
  const request = asRecord(data.request)
  return {
    taskId: String(data.taskId ?? ''),
    asrSubId: typeof data.asrSubId === 'string' ? data.asrSubId : undefined,
    status: stringOr(data.status, 'partial'),
    asr: normalizeAsrEval(data.asr) ?? undefined,
    request: {
      model: stringOr(request.model, 'default'),
      language: typeof request.language === 'string' ? request.language : undefined,
      referenceText: typeof request.referenceText === 'string' ? request.referenceText : undefined,
      batchId: typeof request.batchId === 'string' ? request.batchId : undefined,
      batchItemId: typeof request.batchItemId === 'string' ? request.batchItemId : undefined,
    },
    createdAt: firstString(data, ['createdAt', 'created_at']),
  }
}

function normalizeTaskStatus(payload: unknown): TaskStatusResponse {
  const data = asRecord(payload)
  const cloneTask = asRecord(data.cloneTask)
  const asrTask = asRecord(data.asrTask)

  const normalizeAsrTask = (value: unknown) => {
    const task = asRecord(value)
    return {
      ...(task as unknown as NonNullable<TaskStatusResponse['asrTask']>),
      asrResult:
        task.asrResult === null || task.asrResult === undefined
          ? task.asrResult
          : normalizeAsrResult(task.asrResult),
    }
  }
  const normalizeCloneTask = (value: unknown) => {
    const task = asRecord(value)
    return {
      ...(task as unknown as NonNullable<TaskStatusResponse['cloneTask']>),
      cloneResult:
        task.cloneResult === null || task.cloneResult === undefined
          ? task.cloneResult
          : normalizeCloneResult(task.cloneResult),
    }
  }

  return {
    ...(data as unknown as TaskStatusResponse),
    asrResult:
      data.asrResult === null || data.asrResult === undefined
        ? data.asrResult
        : normalizeAsrResult(data.asrResult),
    cloneResult:
      data.cloneResult === null || data.cloneResult === undefined
        ? data.cloneResult
        : normalizeCloneResult(data.cloneResult),
    cloneTask:
      data.cloneTask === null || data.cloneTask === undefined
        ? data.cloneTask
        : normalizeCloneTask(cloneTask),
    asrTask:
      data.asrTask === null || data.asrTask === undefined
        ? data.asrTask
        : normalizeAsrTask(asrTask),
    asrTasks: Array.isArray(data.asrTasks) ? data.asrTasks.map(normalizeAsrTask) : undefined,
    cloneTasks: Array.isArray(data.cloneTasks) ? data.cloneTasks.map(normalizeCloneTask) : undefined,
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
    const asrResults = Array.isArray(data.asrResults) ? data.asrResults.map(normalizeAsrResult) : undefined
    const asrEval = normalizeAsrEval(data.asrEval ?? asrResults?.at(-1)?.asr ?? data.asr)
    const latestCloneEval = normalizeCloneEval(data.cloneEval) ?? cloneResults?.at(-1)?.cloneEval ?? null
    const protectionEvaluation = normalizeProtectionEvaluation(data.protectionEvaluation ?? details.protectionEvaluation)
    return {
      ...(data as unknown as TaskResult),
      score: canonicalTaskScore(numberOrNull(data.score) ?? numberOrNull(asRecord(data.summary).score), protectionEvaluation),
      originalAudio: normalizeAudio(originalAudio, stringOr(originalAudio.filename, 'original.wav')),
      protectedAudio: normalizeAudio(protectedAudio, stringOr(protectedAudio.filename, 'protected.wav')),
      elapsedSec: normalizeElapsedSec(data),
      perturbation: normalizePerturbation(data.perturbation ?? asRecord(details.perception)),
      protectionQuality: normalizeProtectionQuality(data.protectionQuality ?? data.quality ?? asRecord(details.perception)),
      protectionEvaluation,
      psychoacoustic: normalizePsychoacoustic(data.psychoacoustic ?? asRecord(details.perception)),
      lossFinal: normalizeLossFinal(data.lossFinal ?? generation.lossFinal),
      lossWeights: normalizeLossWeights(data.lossWeights ?? generation.lossWeights),
      optimizationTrace: optimizationTrend,
      averageStepSec: firstNumber(data, ['averageStepSec', 'average_step_sec']) ?? firstNumber(generation, ['averageStepSec', 'average_step_sec']),
      selectedStep: firstNumber(data, ['selectedStep', 'selected_step']) ?? firstNumber(generation, ['selectedStep', 'selected_step']),
      effectiveConfig: asRecord(data.effectiveConfig ?? data.effective_config ?? generation.effectiveConfig ?? generation.effective_config),
      presetName: stringOr(data.presetName ?? data.preset_name ?? generation.presetName ?? generation.preset_name, '') || null,
      asrEval,
      semanticEval: normalizeSemanticEval(data.semanticEval ?? asRecord(details).semantic),
      asrResults,
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
  const rawScore = numberOrNull(summary.score)
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
  const asrResults = Array.isArray(data.asrResults) ? data.asrResults.map(normalizeAsrResult) : undefined
  const asrEval = normalizeAsrEval(asrResults?.at(-1)?.asr ?? detailAsr)
  const asrHasResult = asrEval !== null
  const cloneEval = normalizeCloneEval(data.cloneEval) ?? cloneResults?.at(-1)?.cloneEval ?? null
  const protectionEvaluation = normalizeProtectionEvaluation(data.protectionEvaluation ?? details.protectionEvaluation)
  const score = canonicalTaskScore(rawScore, protectionEvaluation)
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
    inputSource: '已上传音频',
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
    protectionEvaluation,
    psychoacoustic,
    lossFinal,
    lossWeights,
    optimizationTrace: optimizationTrend,
    averageStepSec: firstNumber(data, ['averageStepSec', 'average_step_sec']) ?? firstNumber(detailGeneration, ['averageStepSec', 'average_step_sec']) ?? null,
    asrEval,
    semanticEval: normalizeSemanticEval(data.semanticEval ?? asRecord(details).semantic),
    asrResults,
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
      directDistance: firstNumber(detailSpeaker, ['directDistance', 'direct_distance', 'embeddingDistance', 'embeddingDistanceAfter']),
      directIdentityScore: firstNumber(detailSpeaker, ['directIdentityScore', 'direct_identity_score']),
      scoreStatus: firstString(detailSpeaker, ['scoreStatus', 'score_status']),
      scoreReason: firstString(detailSpeaker, ['scoreReason', 'score_reason']),
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
    const [capabilitiesResult, configResult] = await Promise.allSettled([
      http.get('/api/capabilities'),
      http.get('/api/config'),
    ])
    if (capabilitiesResult.status === 'rejected' && configResult.status === 'rejected') {
      throw capabilitiesResult.reason
    }
    const capabilities = capabilitiesResult.status === 'fulfilled'
      ? asRecord(capabilitiesResult.value.data)
      : {}
    const configEnvelope = configResult.status === 'fulfilled'
      ? asRecord(configResult.value.data)
      : {}
    return {
      ...capabilities,
      ...configEnvelope,
      modelTypes: configEnvelope.modelTypes ?? capabilities.modelTypes,
      config: configEnvelope.config ?? capabilities.config,
      protectQueue: configEnvelope.protectQueue ?? capabilities.protectQueue,
      runtimeConcurrency: configEnvelope.runtimeConcurrency ?? capabilities.runtimeConcurrency,
      runtimePerformance: configEnvelope.runtimePerformance ?? capabilities.runtimePerformance,
    } as unknown as CapabilitiesResponse
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
  async createEvaluationBatch(taskId: string, payload: CreateEvaluationBatchRequest): Promise<EvaluationBatch> {
    const response = await http.post(`/api/tasks/${taskId}/evaluation-batches`, payload)
    return response.data as EvaluationBatch
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
  async deleteAsrEvals(taskId: string): Promise<EvaluationDeleteResult> {
    const response = await http.delete(`/api/tasks/${taskId}/asr-evals`)
    return response.data
  },
  async deleteCloneVoices(taskId: string): Promise<EvaluationDeleteResult> {
    const response = await http.delete(`/api/tasks/${taskId}/clone-voices`)
    return response.data
  },
  async downloadProtectedAudio(taskId: string) {
    const response = await http.get(`/api/tasks/${taskId}/download?type=protected_audio`, {
      responseType: 'blob',
    })
    return {
      blob: response.data,
      filename: filenameFromContentDisposition(response.headers['content-disposition'], 'protected_voice.wav'),
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
