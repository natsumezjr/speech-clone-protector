import { filenameFromContentDisposition } from './contentDisposition.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

const unicodeName = 'record_this_is_the……competition（1）_protected.wav'
const header = `attachment; filename="record_this_is_the_competition_1_protected.wav"; filename*=UTF-8''${encodeURIComponent(unicodeName)}`
assert(filenameFromContentDisposition(header, 'fallback.wav') === unicodeName, 'UTF-8 filename* must take priority')
assert(filenameFromContentDisposition('attachment; filename="plain.wav"', 'fallback.wav') === 'plain.wav', 'plain filename must be supported')
assert(filenameFromContentDisposition(undefined, 'fallback.wav') === 'fallback.wav', 'missing header must use fallback')
