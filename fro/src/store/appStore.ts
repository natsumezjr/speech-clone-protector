import { create } from 'zustand'
import { dataMode, type DataMode } from '@/config/runtime'

export type ToastKind = 'success' | 'error' | 'info'

export interface ToastMessage {
  id: number
  kind: ToastKind
  title: string
  description?: string
}

interface AppState {
  dataMode: DataMode
  toasts: ToastMessage[]
  pushToast: (toast: Omit<ToastMessage, 'id'>) => void
  removeToast: (id: number) => void
}

export const useAppStore = create<AppState>((set) => ({
  dataMode,
  toasts: [],
  pushToast: (toast) =>
    set((state) => ({
      toasts: [...state.toasts, { ...toast, id: Date.now() + Math.random() }],
    })),
  removeToast: (id) => set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
}))
