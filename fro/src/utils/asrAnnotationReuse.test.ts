import type { AsrEvalResponse, SubtaskStatusSnapshot, TaskStatusResponse } from '../types/task.ts'
import { findReusableAsrAnnotation } from './asrAnnotationReuse.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

function result(
  asrSubId: string,
  model: string,
  language: string,
  createdAt: string,
  originalText = 'original transcript',
  protectedText = 'protected transcript',
): AsrEvalResponse {
  return {
    taskId: 'task_reuse',
    asrSubId,
    status: 'available',
    createdAt,
    request: { model, language },
    asr: { model, language, originalText, protectedText },
  }
}

function snapshot(asrResult: AsrEvalResponse, status = 'completed'): SubtaskStatusSnapshot {
  return {
    status,
    asrSubId: asrResult.asrSubId,
    asrRequest: asrResult.request,
    asrResult,
    createdAt: asrResult.createdAt,
  }
}

function taskStatus(overrides: Partial<TaskStatusResponse>): TaskStatusResponse {
  return {
    taskId: 'task_reuse',
    status: 'completed',
    progress: 1,
    stage: 'report_generation',
    message: 'completed',
    createdAt: '2026-08-12T00:00:00+08:00',
    updatedAt: '2026-08-12T00:00:00+08:00',
    error: null,
    ...overrides,
  }
}

const base = result('asr_base', 'openai-whisper:base', 'en', '2026-08-12T01:00:00+08:00')
const mediumOlder = result('asr_medium_old', 'openai-whisper:medium', 'en', '2026-08-12T02:00:00+08:00')
const mediumLatest = result('asr_medium_latest', 'openai/whisper-medium', 'en-US', '2026-08-12T03:00:00+08:00')
const selected = findReusableAsrAnnotation(
  taskStatus({ asrTasks: [snapshot(base), snapshot(mediumOlder), snapshot(mediumLatest)] }),
  'openai-whisper:medium',
  'en',
)
assert(selected?.asrSubId === 'asr_medium_latest', 'The newest complete matching Medium result must be reused')

const legacy = result('asr_legacy', 'whisper:medium', 'english', '2026.8.12 4:05:06')
const selectedLegacy = findReusableAsrAnnotation(
  taskStatus({ asrTask: snapshot(legacy) }),
  'openai-whisper:medium',
  'en',
)
assert(selectedLegacy?.asrSubId === 'asr_legacy', 'The legacy asrTask snapshot must remain reusable')

const topLevel = result('asr_top_level', 'openai-whisper:medium', 'en', '2026-08-12T05:00:00+08:00')
assert(
  findReusableAsrAnnotation(taskStatus({ asrResult: topLevel }), 'openai-whisper:medium', 'en')?.asrSubId === 'asr_top_level',
  'The top-level ASR result must remain reusable',
)

const wrongLanguage = result('asr_wrong_language', 'openai-whisper:medium', 'zh-cn', '2026-08-12T06:00:00+08:00')
assert(
  findReusableAsrAnnotation(taskStatus({ asrTasks: [snapshot(wrongLanguage)] }), 'openai-whisper:medium', 'en') === null,
  'An ASR result from another language must not be reused',
)

const incomplete = result('asr_incomplete', 'openai-whisper:medium', 'en', '2026-08-12T07:00:00+08:00', 'original', '')
const failed = result('asr_failed', 'openai-whisper:medium', 'en', '2026-08-12T08:00:00+08:00')
assert(
  findReusableAsrAnnotation(
    taskStatus({ asrTasks: [snapshot(incomplete), snapshot(failed, 'failed')] }),
    'openai-whisper:medium',
    'en',
  ) === null,
  'Incomplete and failed ASR results must not be reused',
)

const completeSmall = result('asr_small_complete', 'openai/whisper-small', 'en', '2026-08-12T09:00:00+08:00')
const completeWav2Vec = result('asr_wav2vec_complete', 'facebook/wav2vec2-base-960h', 'en', '2026-08-12T10:00:00+08:00')
const fallback = findReusableAsrAnnotation(
  taskStatus({ asrTasks: [snapshot(incomplete), snapshot(completeSmall), snapshot(completeWav2Vec)] }),
  'openai-whisper:medium',
  'en',
)
assert(
  fallback?.asrSubId === 'asr_wav2vec_complete',
  'When matching Medium results are incomplete, the newest complete same-language ASR result must be reused',
)
