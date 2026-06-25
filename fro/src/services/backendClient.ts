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

function normalizeLossTrendPoint(value: unknown, fallbackStep: number): TaskResult['charts']['optimizationTrend'][number] | null {
  const point = asRecord(value)
  const step = firstNumber(point, ['step', 'epoch', 'iteration', 'iter']) ?? fallbackStep
  const normalized = {
    step,
    Lfeat: firstNumber(point, ['Lfeat', 'Lfea', 'lossFeature', 'loss_timbre', 'L_feature']),
    Lsem: firstNumber(point, ['Lsem', 'lossSemantic', 'loss_semantic', 'L_semantic']),
    Lpsy: firstNumber(point, ['Lpsy', 'lossPsy', 'loss_psy', 'L_psy']),
    L2: firstNumber(point, ['L2', 'lossL2', 'loss_l2', 'l2Norm']),
    total: firstNumber(point, ['total', 'lossTotal', 'loss_total']),
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

function nearestLossPoint(points: TaskResult['charts']['optimizationTrend'], step: number) {
  return points.reduce((best, point) => (Math.abs(point.step - step) < Math.abs(best.step - step) ? point : best), points[0])
}

function uniqueLossPoints(points: TaskResult['charts']['optimizationTrend']) {
  const seen = new Set<number>()
  return points.filter((point) => {
    if (seen.has(point.step)) return false
    seen.add(point.step)
    return true
  })
}

function sampleLossTrend(points: TaskResult['charts']['optimizationTrend']): TaskResult['charts']['optimizationTrend'] {
  if (points.length <= 10) return points
  const ordered = [...points].sort((a, b) => a.step - b.step)
  const maxStep = Math.max(...ordered.map((point) => point.step), ordered.length)
  if (maxStep <= 10) return ordered
  if (maxStep % 10 === 0) {
    const interval = maxStep / 10
    return uniqueLossPoints(Array.from({ length: 10 }, (_, index) => nearestLossPoint(ordered, interval * (index + 1))))
  }
  if (maxStep < 20) {
    return uniqueLossPoints([...ordered.slice(0, 7), ...ordered.slice(-3)])
  }
  const sampled = Array.from({ length: 10 }, (_, index) => {
    const position = Math.round((index / 9) * (ordered.length - 1))
    return ordered[position]
  })
  return uniqueLossPoints(sampled)
}

function normalizeLossFinal(value: unknown): NonNullable<TaskResult['generation']>['lossFinal'] {
  const record = asRecord(value)
  return {
    Lfeat: firstNumber(record, ['Lfeat', 'Lfea', 'lossFeature', 'loss_timbre', 'L_feature']),
    Lsem: firstNumber(record, ['Lsem', 'lossSemantic', 'loss_semantic', 'L_semantic']),
    Lpsy: firstNumber(record, ['Lpsy', 'lossPsy', 'loss_psy', 'L_psy']),
    L2: firstNumber(record, ['L2', 'lossL2', 'loss_l2', 'l2Norm']),
    total: firstNumber(record, ['total', 'lossTotal', 'loss_total']),
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
  const filename = stringOr(data.filename, fallbackName)
  const rawSrc = typeof data.src === 'string' ? data.src : undefined
  const rawAudioUrl = typeof data.audioUrl === 'string' ? data.audioUrl : rawSrc
  const rawDownloadUrl = typeof data.downloadUrl === 'string' ? data.downloadUrl : undefined
  const rawObjectUrl = typeof data.objectUrl === 'string' ? data.objectUrl : undefined
  return {
    fileId: typeof data.fileId === 'string' ? data.fileId : undefined,
    filename,
    durationSec: numberOrNull(data.durationSec ?? data.duration) ?? undefined,
    duration: numberOrNull(data.duration ?? data.durationSec) ?? undefined,
    sampleRate: numberOrNull(data.sampleRate) ?? undefined,
    channels: numberOrNull(data.channels) ?? undefined,
    bitDepth: numberOrNull(data.bitDepth) ?? undefined,
    sizeBytes: numberOrNull(data.sizeBytes) ?? 0,
    format: stringOr(data.format, filename.split('.').pop()?.toUpperCase() ?? 'AUDIO'),
    src: absoluteUrl(rawSrc),
    audioUrl: absoluteUrl(rawAudioUrl),
    downloadUrl: absoluteUrl(rawDownloadUrl),
    objectUrl: rawObjectUrl,
    uploadedAt: typeof data.uploadedAt === 'string' ? data.uploadedAt : undefined,
    fingerprint: typeof data.fingerprint === 'string' ? data.fingerprint : undefined,
  }
}

function normalizeCloneResult(payload: unknown): CloneVoiceResult {
  const data = asRecord(payload)
  const request = asRecord(data.request)
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
    return {
      ...(data as unknown as TaskResult),
      originalAudio: normalizeAudio(originalAudio, stringOr(originalAudio.filename, 'original.wav')),
      protectedAudio: normalizeAudio(protectedAudio, stringOr(protectedAudio.filename, 'protected.wav')),
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
    elapsedSec: numberOrNull(data.elapsedSec),
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
      lossFinal: normalizeLossFinal(detailGeneration.lossFinal),
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
    return response.data
  },
  async deleteTask(taskId: string): Promise<void> {
    await http.delete(`/api/tasks/${taskId}`)
  },
  async downloadProtectedAudio(taskId: string) {
    const response = await http.get(`/api/tasks/${taskId}/download/protected-audio`, {
      responseType: 'blob',
    })
    return {
      blob: response.data,
      filename: filenameFromDisposition(response.headers['content-disposition'], 'protected_voice.wav'),
    }
  },
  async exportReport(taskId: string): Promise<Blob> {
    const response = await http.post('/api/reports/export', { taskId }, { responseType: 'blob' })
    return response.data
  },
  async exportCsv(taskId: string): Promise<Blob> {
    const response = await http.get(`/api/tasks/${taskId}/export/csv`, { responseType: 'blob' })
    return response.data
  },
  async downloadEvidenceZip(taskId: string): Promise<Blob> {
    const response = await http.get(`/api/tasks/${taskId}/download/evidence`, { responseType: 'blob' })
    return response.data
  },
}
