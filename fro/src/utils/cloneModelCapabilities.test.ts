import {
  cloneModelRequiresReferenceText,
  normalizeCloneReferenceTextRequest,
} from './cloneModelCapabilities.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

assert(
  cloneModelRequiresReferenceText({ requiresReferenceText: true, promptRequired: false }),
  'The backend-owned requiresReferenceText field must take precedence over the legacy alias',
)
assert(
  !cloneModelRequiresReferenceText({ requiresReferenceText: false, promptRequired: true }),
  'An explicit backend-owned false value must not be overridden by the legacy alias',
)
assert(
  cloneModelRequiresReferenceText({ promptRequired: true }),
  'Legacy promptRequired capabilities must remain compatible',
)
assert(
  !cloneModelRequiresReferenceText({}),
  'Models without a reference-text requirement must default to false',
)

const stripped = normalizeCloneReferenceTextRequest(
  {
    text: 'clone text',
    model: 'XTTS-v2',
    annotationSource: 'asr',
    annotationAsrSubId: 'asr_should_not_be_sent',
    speakerPrompt: 'unused prompt',
    originalSpeakerPrompt: 'unused original prompt',
    protectedSpeakerPrompt: 'unused protected prompt',
  },
  { requiresReferenceText: false },
)
assert(stripped.annotationSource === undefined, 'Annotation source must be cleared for models that do not need reference text')
assert(stripped.annotationAsrSubId === undefined, 'ASR annotation id must be cleared for models that do not need reference text')
assert(stripped.speakerPrompt === undefined, 'Manual annotation must be cleared for models that do not need reference text')
assert(stripped.originalSpeakerPrompt === undefined, 'Original ASR text must be cleared for models that do not need reference text')
assert(stripped.protectedSpeakerPrompt === undefined, 'Protected ASR text must be cleared for models that do not need reference text')

const required = normalizeCloneReferenceTextRequest(
  {
    text: 'clone text',
    model: 'CosyVoice2-0.5B',
    speakerPrompt: 'manual annotation',
  },
  { requiresReferenceText: true },
)
assert(required.annotationSource === 'manual', 'Required reference text must default to the manual source')
assert(required.speakerPrompt === 'manual annotation', 'Required manual annotation must be retained')
