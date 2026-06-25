import type { ApiErrorPayload, TaskStatusResponse } from '@/types/task'

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {}
}

function nonEmptyString(value: unknown) {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function clip(value: string, max = 180) {
  return value.length > max ? `${value.slice(0, max - 1)}…` : value
}

export function formatStructuredApiError(error: ApiErrorPayload | null | undefined, fallback = '请求失败。') {
  if (!error) return fallback
  const details = asRecord(error.details)
  const capabilities = asRecord(details.capabilities)
  const protectCapability = asRecord(capabilities.protect_generation)
  const reason = nonEmptyString(details.reason) ?? nonEmptyString(protectCapability.reason)
  const suggestion = nonEmptyString(details.suggestion)

  const lines = [error.message || fallback]
  if (error.stage) lines.push(`阶段：${error.stage}`)
  if (error.requestId) lines.push(`RequestId：${error.requestId}`)
  if (reason) lines.push(`原因：${clip(reason)}`)
  if (suggestion) lines.push(`建议：${clip(suggestion)}`)
  return lines.join('\n')
}

export function formatTaskFailure(status: TaskStatusResponse) {
  const error = status.error
  if (typeof error === 'string' && error.trim()) return clip(error.trim(), 280)
  return formatStructuredApiError(error as ApiErrorPayload | null, status.message || '任务执行失败')
}
