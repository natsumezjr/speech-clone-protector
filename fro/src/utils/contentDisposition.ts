export function filenameFromContentDisposition(header: string | undefined, fallback: string) {
  if (!header) return fallback
  const encodedMatch = /filename\*=UTF-8''([^;]+)/i.exec(header)
  if (encodedMatch) {
    try {
      return decodeURIComponent(encodedMatch[1].trim().replace(/^"|"$/g, ''))
    } catch {
      // Fall through to the ASCII filename when percent encoding is malformed.
    }
  }
  const plainMatch = /filename="?([^";]+)"?/i.exec(header)
  return plainMatch ? plainMatch[1].trim() : fallback
}
