import { create } from 'zustand'

export type ToastKind = 'success' | 'error' | 'info'

export interface ToastMessage {
  id: string
  kind: ToastKind
  title: string
  description?: string
}

interface AppState {
  toasts: ToastMessage[]
  pushToast: (toast: Omit<ToastMessage, 'id'> & { id?: string; dedupeMs?: number }) => void
  removeToast: (id: string) => void
}

const recentToasts = new Map<string, number>()

export const useAppStore = create<AppState>((set) => ({
  toasts: [],
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