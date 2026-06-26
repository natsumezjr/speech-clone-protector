import axios from 'axios'
import { apiBaseUrl } from '@/config/runtime'
import type { ApiClient } from '@/types/api'
import type { AudioFileMeta } from '@/types/audio'
import type { ApiErrorPayload, AsrEvalRequest, AsrEvalResponse, CloneVoiceRequest, CloneVoiceResult, HistoryTask, ProtectionTaskRequest, TaskDetailsResponse, TaskResult, TaskStatusResponse } from '@/types/task'
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
  const normalized = {
    step,
    Lfeat: firstNumber(point, ['Lfeat', 'lFeat', 'Lfea', 'lossFeature', 'loss_timbre', 'L_feature']),
    Lsem: firstNumber(point, ['Lsem', 'lSem', 'lossSemantic', 'loss_semantic', 'L_semantic']),
    Lpsy: firstNumber(point, ['Lpsy', 'lPsy', 'lossPsy', 'loss_psy', 'L_psy']),
    L2: firstNumber(point, ['L2', 'l2', 'lossL2', 'loss_l2', 'l2Norm']),
    total: firstNumber(point, ['total', 'totalLoss', 'lossTotal', 'loss_total']),
    stepElapsedSec: firstNumber(point, ['stepElapsedSec', 'step_elapsed_sec', 'elapsedSec', 'elapsed']),
  }
  const hasLoss = [normalized.Lfeat, normalized.Lsem, normalized.Lpsy, normalized.L2, normalized.total].some((item) => item !== null)
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
  return {
    Lfeat: firstNumber(record, ['Lfeat', 'lFeat', 'Lfea', 'lossFeature', 'loss_timbre', 'L_feature']),
    Lsem: firstNumber(record, ['Lsem', 'lSem', 'lossSemantic', 'loss_semantic', 'L_semantic']),
    Lpsy: firstNumber(record, ['Lpsy', 'lPsy', 'lossPsy', 'loss_psy', 'L_psy']),
    L2: firstNumber(record, ['L2', 'l2', 'lossL2', 'loss_l2', 'l2Norm']),
    total: firstNumber(record, ['total', 'totalLoss', 'lossTotal', 'loss_total']),
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
    maskingThreshold: Array.isArray(record.maskingThreshold) ? (record.maskingThreshold as NonNullable<TaskResult['psychoacoustic']>['maskingThreshold']) : null,
    perturbationSpectrum: Array.isArray(record.perturbationSpectrum) ? (record.perturbationSpectrum as NonNullable<TaskResult['psychoacoustic']>['perturbationSpectrum']) : null,
  }
}

function normalizeLossWeights(value: unknown): TaskResult['lossWeights'] {
  const record = asRecord(value)
  return {
    lambdaFeat: firstNumber(record, ['lambdaFeat', 'weightFeature', 'weight_feature']),
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
  const originalText = firstString(data, ['originalText', 'original_transcript', 'referenceText', 'cleanTranscription', 'beforeText'])
  const protectedText = firstString(data, ['protectedText', 'protected_transcript', 'protectedTranscription', 'afterText'])
  const hasMetric = hasAnyNumber(data, ['wer', 'WER', 'cer', 'CER', 'insertRate', 'ir', 'insertionRate', 'deleteRate', 'dr', 'deletionRate', 'substituteRate', 'sr', 'substitutionRate', 'tokenErrorRate', 'token_error_rate', 'tokenChangeRate', 'token_change_rate', 'semanticDrift', 'semantic_drift'])
  if (status && !['computed', 'partial', 'completed', 'success', 'finished'].includes(status) && !originalText && !protectedText && !hasMetric) return null
  if (!status && !originalText && !protectedText && !hasMetric) return null
  return {
    model: firstString(data, ['model', 'asrModel', 'asr_model']) ?? undefined,
    asrModel: firstString(data, ['asrModel', 'asr_model', 'model']),
    language: firstString(data, ['language', 'lang']),
    originalText,
    protectedText,
    wer: firstNumber(data, ['wer', 'WER']),
    cer: firstNumber(data, ['cer', 'CER']),
    substituteRate: firstNumber(data, ['substituteRate', 'sr', 'substitutionRate']),
    insertRate: firstNumber(data, ['insertRate', 'ir', 'insertionRate']),
    deleteRate: firstNumber(data, ['deleteRate', 'dr', 'deletionRate']),
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
    cloneRadar: Array.isArray(data.cloneRadar) ? (data.cloneRadar as NonNullable<TaskResult['cloneEval']>['cloneRadar']) : null,
    cloneTrend: Array.isArray(data.cloneTrend) ? (data.cloneTrend as NonNullable<TaskResult['cloneEval']>['cloneTrend']) : null,
    cloneDefenseScore: firstNumber(data, ['cloneDefenseScore', 'clone_defense_score']),
    createdAt: firstString(data, ['createdAt', 'created_at']),
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
      averageStepSec: firstNumber(data, ['averageStepSec', 'average_step_sec']),
      asrEval,
      cloneEval: latestCloneEval,
      generation: {
        ...(asRecord(data.generation) as NonNullable<TaskResult['generation']>),
        lossFinal: normalizeLossFinal(generation.lossFinal),
        optimizationTrace: normalizeLossTrend(generation.optimizationTrace),
        steps: numberOrNull(generation.steps),
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
  const detailSemantic = asRecord(details.semantic)
  const backend = asRecord(data.backend)
  const charts = asRecord(data.charts)
  const optimizationTrend = normalizeLossTrend(charts.optimizationTrend ?? detailGeneration.optimizationTrace ?? charts.trend)
  const originalRaw = asRecord(audio.original)
  const protectedRaw = asRecord(audio.protected)
  const metricSources = asRecord(summary.metricSources)
  const score = numberOrNull(summary.score)
  const snr = numberOrNull(primary.snr)
  const pesq = numberOrNull(primary.pesq)
  const simAfter = numberOrNull(primary.speakerSimilarity ?? detailSpeaker.simOriginalProtected)
  const simBefore = null
  const originalText = optionalString(detailAsr.referenceText ?? detailAsr.cleanTranscription)
  const protectedText = optionalString(detailAsr.protectedTranscription)
  const cloneResults = Array.isArray(data.cloneResults) ? data.cloneResults.map(normalizeCloneResult) : undefined
  const lossFinal = normalizeLossFinal(detailGeneration.lossFinal)
  const lossWeights = normalizeLossWeights(detailGeneration.lossWeights)
  const elapsedSec = normalizeElapsedSec(data)
  const asrEval = normalizeAsrEval(detailAsr)
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
    averageStepSec: firstNumber(data, ['averageStepSec', 'average_step_sec']) ?? null,
    asrEval,
    cloneEval,
    cloneResults,
    asr: {
      originalText,
      protectedText,
      wer: optionalNumber(primary.wer) ?? optionalNumber(detailAsr.wer),
      cer: optionalNumber(primary.cer) ?? optionalNumber(detailAsr.cer),
      tokenErrorRate: optionalNumber(primary.tokenErrorRate) ?? optionalNumber(detailSemantic.tokenErrorRate),
      semanticDrift: optionalNumber(primary.semanticDrift) ?? optionalNumber(detailSemantic.semanticDrift),
      insertRate: optionalNumber(asRecord(detailAsr.breakdown).insertRate),
      deleteRate: optionalNumber(asRecord(detailAsr.breakdown).deleteRate),
      substituteRate: optionalNumber(asRecord(detailAsr.breakdown).substituteRate),
      status: typeof detailAsr.status === 'string' ? detailAsr.status : undefined,
    },
    speaker: {
      simBefore,
      simAfter,
      simDropRate: optionalNumber(asRecord(details.downstreamTts).simDropRate),
      embeddingDistanceBefore: null,
      embeddingDistanceAfter: optionalNumber(detailSpeaker.embeddingDistance),
      source: typeof asRecord(metricSources.speakerSimilarity).source === 'string' ? asRecord(metricSources.speakerSimilarity).source as string : undefined,
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
    generation: {
      lossFinal,
      optimizationTrace: normalizeLossTrend(detailGeneration.optimizationTrace),
      steps: numberOrNull(detailGeneration.steps),
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
    ...data,
    elapsedSec: data.asrElapsedSec,
    startedAt: data.asrStartedAt,
    completedAt: data.asrCompletedAt,
  })
  const cloneElapsedSec = normalizeElapsedSec({
    ...data,
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
  async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    const response = await http.get(`/api/tasks/${taskId}/status`)
    return response.data
  },
  async getTaskResult(taskId: string): Promise<TaskResult> {
    const response = await http.get(`/api/tasks/${taskId}/result`)
    return normalizeTaskResult(response.data)
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
