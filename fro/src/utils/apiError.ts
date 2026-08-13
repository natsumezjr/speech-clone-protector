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

const stageLabels: Record<string, string> = {
  api: '请求处理',
  file_preprocess: '音频准备',
  protect_generation: '生成保护音频',
  report_generation: '整理保护结果',
  asr_eval: '语音识别测试',
  downstream_tts_eval: '语音克隆测试',
}

export function formatStructuredApiError(error: ApiErrorPayload | null | undefined, fallback = '请求失败。') {
  if (!error) return fallback
  const details = asRecord(error.details)
  const capabilities = asRecord(details.capabilities)
  const protectCapability = asRecord(capabilities.protect_generation)
  const reason = nonEmptyString(details.reason) ?? nonEmptyString(protectCapability.reason)
  const suggestion = nonEmptyString(details.suggestion)

  const lines = [error.message || fallback]
  if (error.stage) lines.push(`处理环节：${stageLabels[error.stage] ?? '任务处理'}`)
  if (error.requestId) lines.push(`问题编号：${error.requestId}`)
  if (reason) lines.push(`原因：${clip(reason)}`)
  if (suggestion) lines.push(`建议：${clip(suggestion)}`)
  return lines.join('\n')
}

export function formatTaskFailure(status: TaskStatusResponse) {
  const error = status.error
  if (typeof error === 'string' && error.trim()) return clip(error.trim(), 280)
  return formatStructuredApiError(error as ApiErrorPayload | null, status.message || '任务执行失败')
}
