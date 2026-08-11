import type { CloneVoiceRequest, RuntimeModelOption } from '@/types/task'

export type CloneModelCapability = Pick<RuntimeModelOption, 'requiresReferenceText' | 'promptRequired'>

export function cloneModelRequiresReferenceText(model?: CloneModelCapability | null) {
  if (typeof model?.requiresReferenceText === 'boolean') return model.requiresReferenceText
  return model?.promptRequired === true
}

export function withoutCloneReferenceText(request: CloneVoiceRequest): CloneVoiceRequest {
  return {
    ...request,
    annotationSource: undefined,
    annotationAsrSubId: undefined,
    annotationAsrModel: undefined,
    annotationCreatedAt: undefined,
    speakerPrompt: undefined,
    originalSpeakerPrompt: undefined,
    protectedSpeakerPrompt: undefined,
  }
}

export function normalizeCloneReferenceTextRequest(
  request: CloneVoiceRequest,
  model?: CloneModelCapability | null,
): CloneVoiceRequest {
  if (!cloneModelRequiresReferenceText(model)) return withoutCloneReferenceText(request)
  if (request.annotationSource === 'asr') return request
  return {
    ...request,
    annotationSource: 'manual',
    annotationAsrSubId: undefined,
    annotationAsrModel: undefined,
    annotationCreatedAt: undefined,
    originalSpeakerPrompt: undefined,
    protectedSpeakerPrompt: undefined,
  }
}
