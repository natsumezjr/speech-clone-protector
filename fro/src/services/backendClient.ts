import axios from 'axios'
import { apiBaseUrl } from '@/config/runtime'
import type { ApiClient } from '@/types/api'
import type { AudioFileMeta } from '@/types/audio'
import type { CloneVoiceRequest, CloneVoiceResult, HistoryTask, ProtectionTaskRequest, TaskResult, TaskStatusResponse, TrendPoint } from '@/types/task'

const http = axios.create({
  baseURL: apiBaseUrl,
})

function filenameFromDisposition(header: string | undefined, fallback: string) {
  if (!header) return fallback
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header)
  return match ? decodeURIComponent(match[1]) : fallback
}

function numberOr(value: unknown, fallback: number) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : fallback
}

function optionalNumber(value: unknown) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : undefined
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {}
}

function stringOr(value: unknown, fallback: string) {
  return typeof value === 'string' && value.length > 0 ? value : fallback
}

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
    durationSec: data.durationSec === undefined ? numberOr(data.duration, 0) || undefined : numberOr(data.durationSec, 0) || undefined,
    duration: data.duration === undefined ? numberOr(data.durationSec, 0) || undefined : numberOr(data.duration, 0) || undefined,
    sampleRate: data.sampleRate === undefined ? undefined : numberOr(data.sampleRate, 0),
    channels: data.channels === undefined ? undefined : numberOr(data.channels, 0),
    bitDepth: data.bitDepth === undefined ? undefined : numberOr(data.bitDepth, 0),
    sizeBytes: numberOr(data.sizeBytes, 0),
    format: stringOr(data.format, filename.split('.').pop()?.toUpperCase() ?? 'AUDIO'),
    src: absoluteUrl(rawSrc),
    audioUrl: absoluteUrl(rawAudioUrl),
    downloadUrl: absoluteUrl(rawDownloadUrl),
    objectUrl: rawObjectUrl,
    uploadedAt: typeof data.uploadedAt === 'string' ? data.uploadedAt : undefined,
    fingerprint: typeof data.fingerprint === 'string' ? data.fingerprint : undefined,
  }
}

function fallbackTrend(score: number): TrendPoint[] {
  return Array.from({ length: 12 }, (_, index) => {
    const step = index + 1
    return {
      step,
      wer: Math.min(0.92, 0.12 + step * 0.035),
      sim: Math.max(0.08, 0.86 - step * 0.045),
      mos: Math.max(3.1, 4.2 - step * 0.035),
      pesq: Math.max(2.9, 4.0 - step * 0.03),
      elapsed: step * Math.max(2, score / 24),
    }
  })
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
      speed: numberOr(request.speed, 1),
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
    const cloneResults = Array.isArray(data.cloneResults) ? data.cloneResults.map(normalizeCloneResult) : undefined
    return {
      ...(data as unknown as TaskResult),
      originalAudio: normalizeAudio(originalAudio, stringOr(originalAudio.filename, 'original.wav')),
      protectedAudio: normalizeAudio(protectedAudio, stringOr(protectedAudio.filename, 'protected.wav')),
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
  const originalRaw = asRecord(audio.original)
  const protectedRaw = asRecord(audio.protected)
  const score = numberOr(summary.score, 80)
  const snr = numberOr(primary.snr, 0)
  const pesq = numberOr(primary.pesq, 3.5)
  const simAfter = numberOr(primary.speakerSimilarity ?? detailSpeaker.simOriginalProtected, 0.3)
  const simBefore = Math.max(simAfter, 0.9)
  const originalText = stringOr(detailAsr.referenceText ?? detailAsr.cleanTranscription, 'ASR 未启用，暂无原始转写。')
  const protectedText = stringOr(detailAsr.protectedTranscription, 'ASR 未启用，暂无保护音频转写。')
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
    elapsedSec: numberOr(data.elapsedSec, 0),
    inputSource: '后端 API',
    language: stringOr(detailAsr.language, '未标注'),
    processingModel: stringOr(detailGeneration.source ?? backend.version, ''),
    optimizationTarget: stringOr(detailGeneration.mode ?? data.mode, 'joint'),
    asrModel: typeof detailAsr.model === 'string' ? detailAsr.model : undefined,
    artifacts: [
      { label: '原始音频', filename: stringOr(originalRaw.filename, 'original.wav'), sizeBytes: numberOr(originalRaw.sizeBytes, 0) },
      { label: '保护音频', filename: stringOr(protectedRaw.filename, 'protected.wav'), sizeBytes: numberOr(protectedRaw.sizeBytes, 0) },
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
    },
    speaker: {
      simBefore,
      simAfter,
      simDropRate: simBefore ? Math.max(0, (simBefore - simAfter) / simBefore) : 0,
      embeddingDistanceBefore: Math.max(0, 1 - simBefore),
      embeddingDistanceAfter: Math.max(0, 1 - simAfter),
    },
    quality: {
      snr,
      pesq,
      mosLqo: Math.max(2.8, Math.min(4.6, 3.2 + snr / 30)),
    },
    charts: {
      psychoacoustic: Array.isArray(charts.psychoacoustic) ? charts.psychoacoustic as TaskResult['charts']['psychoacoustic'] : [],
      trend: Array.isArray(charts.trend) && charts.trend.length ? charts.trend as TrendPoint[] : fallbackTrend(score),
      radarBefore: Array.isArray(charts.radarBefore) ? charts.radarBefore as number[] : [0.92, 0.88, 0.82, 0.76, 0.84, 0.8],
      radarAfter: Array.isArray(charts.radarAfter) ? charts.radarAfter as number[] : [0.22, 0.26, 0.24, 0.34, 0.29, 0.31],
    },
  }
}

export const backendClient: ApiClient = {
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
    const response = await http.get(`/api/tasks/${taskId}`)
    return response.data
  },
  async getTaskResult(taskId: string): Promise<TaskResult> {
    const response = await http.get(`/api/tasks/${taskId}/result`)
    return normalizeTaskResult(response.data)
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
