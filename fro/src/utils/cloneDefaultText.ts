export const defaultCloneTextEnglish = "This test shows how VoiceShield protects a speaker's voice."
export const defaultCloneTextChinese = '本测试展示了 VoiceShield 如何保护说话人的声音。'

export function defaultCloneTextForLanguage(language?: string | null) {
  return String(language ?? '').toLowerCase().startsWith('zh')
    ? defaultCloneTextChinese
    : defaultCloneTextEnglish
}

export function translateDefaultCloneText(currentText: string, nextLanguage?: string | null) {
  if (currentText !== defaultCloneTextEnglish && currentText !== defaultCloneTextChinese) {
    return currentText
  }
  return defaultCloneTextForLanguage(nextLanguage)
}
