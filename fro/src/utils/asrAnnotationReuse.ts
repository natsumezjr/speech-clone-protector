import type { AsrEvalResponse, AsrEvalRequest, SubtaskStatusSnapshot, TaskStatusResponse } from '@/types/task'

type AsrAnnotationCandidate = {
  result: AsrEvalResponse
  request?: AsrEvalRequest | null
  status?: string | null
  subId?: string | null
  createdAt?: string | null
  updatedAt?: string | null
  order: number
}

const unusableStatuses = new Set(['queued', 'running', 'failed', 'error', 'cancelled', 'canceled'])

function canonicalAsrModel(value: unknown) {
  const compact = String(value ?? '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '')
  if (!compact) return ''
  return compact.startsWith('openaiwhisper') ? compact.slice('openai'.length) : compact
}

function canonicalLanguage(value: unknown) {
  const normalized = String(value ?? '').trim().toLowerCase().replace('_', '-')
  if (!normalized) return ''
  if (normalized === 'chinese' || normalized.startsWith('zh')) return 'zh'
  if (normalized === 'english' || normalized.startsWith('en')) return 'en'
  return normalized
}

function timestamp(value?: string | null) {
  if (!value) return 0
  const dotted = value.trim().match(/^(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?$/)
  if (dotted) {
    return new Date(
      Number(dotted[1]),
      Number(dotted[2]) - 1,
      Number(dotted[3]),
      Number(dotted[4] ?? 0),
      Number(dotted[5] ?? 0),
      Number(dotted[6] ?? 0),
    ).getTime()
  }
  const parsed = new Date(value).getTime()
  return Number.isFinite(parsed) ? parsed : 0
}

function fromSnapshot(snapshot: SubtaskStatusSnapshot, order: number): AsrAnnotationCandidate | null {
  if (!snapshot.asrResult) return null
  return {
    result: snapshot.asrResult,
    request: snapshot.asrRequest,
    status: snapshot.status,
    subId: snapshot.asrSubId,
    createdAt: snapshot.createdAt,
    updatedAt: snapshot.updatedAt,
    order,
  }
}

export function findReusableAsrAnnotation(
  status: TaskStatusResponse,
  expectedModel: string,
  expectedLanguage: string,
): AsrEvalResponse | null {
  const candidates: AsrAnnotationCandidate[] = []
  let order = 0
  for (const snapshot of status.asrTasks ?? []) {
    const candidate = fromSnapshot(snapshot, order)
    order += 1
    if (candidate) candidates.push(candidate)
  }
  if (status.asrTask) {
    const candidate = fromSnapshot(status.asrTask, order)
    order += 1
    if (candidate) candidates.push(candidate)
  }
  if (status.asrResult) {
    candidates.push({
      result: status.asrResult,
      request: status.asrResult.request,
      status: status.asrResult.status,
      subId: status.asrResult.asrSubId,
      createdAt: status.asrResult.createdAt,
      order,
    })
  }

  const targetModel = canonicalAsrModel(expectedModel)
  const targetLanguage = canonicalLanguage(expectedLanguage)
  const usable = candidates.filter((candidate) => {
    const result = candidate.result
    const asr = result.asr
    const candidateStatus = String(candidate.status ?? result.status ?? '').trim().toLowerCase()
    if (unusableStatuses.has(candidateStatus)) return false
    const subId = String(candidate.subId ?? result.asrSubId ?? '').trim()
    if (!subId) return false
    const originalText = asr?.originalText?.trim() ?? ''
    const protectedText = asr?.protectedText?.trim() ?? ''
    if (!originalText || !protectedText) return false
    const request = result.request ?? candidate.request
    const language = request?.language ?? asr?.language
    return canonicalLanguage(language) === targetLanguage
  })

  const newestFirst = (left: AsrAnnotationCandidate, right: AsrAnnotationCandidate) => {
    const rightTime = timestamp(right.updatedAt ?? right.result.createdAt ?? right.createdAt)
    const leftTime = timestamp(left.updatedAt ?? left.result.createdAt ?? left.createdAt)
    return rightTime - leftTime || right.order - left.order
  }
  const matchingModel = usable.filter((candidate) => {
    const request = candidate.result.request ?? candidate.request
    const model = candidate.result.asr?.model ?? candidate.result.asr?.asrModel ?? request?.model
    return canonicalAsrModel(model) === targetModel
  })
  matchingModel.sort(newestFirst)
  usable.sort(newestFirst)
  const selected = matchingModel[0] ?? usable[0]
  if (!selected) return null
  const asrSubId = String(selected.subId ?? selected.result.asrSubId ?? '').trim()
  return {
    ...selected.result,
    asrSubId,
    request: selected.result.request ?? selected.request ?? undefined,
  }
}
