import { create } from 'zustand'

export type ThemeMode = 'light' | 'dark'
export type ToastKind = 'success' | 'error' | 'info'

export interface ToastMessage {
  id: string
  kind: ToastKind
  title: string
  description?: string
}

interface AppState {
  toasts: ToastMessage[]
  themeMode: ThemeMode
  setThemeMode: (mode: ThemeMode) => void
  toggleThemeMode: () => void
  pushToast: (toast: Omit<ToastMessage, 'id'> & { id?: string; dedupeMs?: number }) => void
  removeToast: (id: string) => void
}

const recentToasts = new Map<string, number>()
const themeStorageKey = 'vguard-theme-mode'

function isThemeMode(value: string | null): value is ThemeMode {
  return value === 'light' || value === 'dark'
}

function readStoredTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'light'
  const storedTheme = window.localStorage.getItem(themeStorageKey)
  return isThemeMode(storedTheme) ? storedTheme : 'light'
}

function applyThemeMode(mode: ThemeMode) {
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = mode
  }
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(themeStorageKey, mode)
  }
}

const initialThemeMode = readStoredTheme()
applyThemeMode(initialThemeMode)

export const useAppStore = create<AppState>((set) => ({
  toasts: [],
  themeMode: initialThemeMode,
  setThemeMode: (mode) =>
    set(() => {
      applyThemeMode(mode)
      return { themeMode: mode }
    }),
  toggleThemeMode: () =>
    set((state) => {
      const mode = state.themeMode === 'dark' ? 'light' : 'dark'
      applyThemeMode(mode)
      return { themeMode: mode }
    }),
  pushToast: (toast) =>
    set((state) => {
      const id = toast.id ?? `${toast.kind}:${toast.title}:${toast.description ?? ''}`
      const now = Date.now()
      const dedupeMs = toast.dedupeMs ?? 2500
      const previous = recentToasts.get(id) ?? 0
      if (now - previous < dedupeMs) return state
      recentToasts.set(id, now)
      const nextToast = { kind: toast.kind, title: toast.title, description: toast.description, id }
      const withoutSameId = state.toasts.filter((item) => item.id !== id)
      return { toasts: [...withoutSameId, nextToast] }
    }),
  removeToast: (id) => set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
}))
