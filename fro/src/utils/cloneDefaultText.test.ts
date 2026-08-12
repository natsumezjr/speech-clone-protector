import {
  defaultCloneTextChinese,
  defaultCloneTextEnglish,
  defaultCloneTextForLanguage,
  translateDefaultCloneText,
} from './cloneDefaultText.ts'

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message)
}

assert(defaultCloneTextForLanguage('zh-cn') === defaultCloneTextChinese, 'zh-cn must use the Chinese default text')
assert(defaultCloneTextForLanguage('zh-CN') === defaultCloneTextChinese, 'Chinese language matching must be case-insensitive')
assert(defaultCloneTextForLanguage('zh') === defaultCloneTextChinese, 'zh must use the Chinese default text')
assert(defaultCloneTextForLanguage('en') === defaultCloneTextEnglish, 'en must use the English default text')
assert(defaultCloneTextForLanguage('en-US') === defaultCloneTextEnglish, 'English variants must use the English default text')
assert(defaultCloneTextForLanguage('auto') === defaultCloneTextEnglish, 'Non-Chinese languages must use the English default text')

assert(
  translateDefaultCloneText(defaultCloneTextEnglish, 'zh-cn') === defaultCloneTextChinese,
  'The English system default must switch to Chinese',
)
assert(
  translateDefaultCloneText(defaultCloneTextChinese, 'en') === defaultCloneTextEnglish,
  'The Chinese system default must switch back to English',
)
assert(
  translateDefaultCloneText('用户自定义文本', 'en') === '用户自定义文本',
  'Custom text must not be overwritten when the language changes',
)
assert(
  translateDefaultCloneText('', 'zh-cn') === '',
  'An intentionally cleared text field must remain empty',
)
assert(
  translateDefaultCloneText(` ${defaultCloneTextEnglish}`, 'zh-cn') === ` ${defaultCloneTextEnglish}`,
  'Text edited around a default value must be treated as custom text',
)
