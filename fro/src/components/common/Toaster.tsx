import { CheckCircle2, Info, X, XCircle } from 'lucide-react'
import { useEffect } from 'react'
import { useAppStore, type ToastKind } from '@/store/appStore'
import { cn } from '@/lib/utils'

const icons: Record<ToastKind, typeof Info> = {
  success: CheckCircle2,
  error: XCircle,
  info: Info,
}

const tones: Record<ToastKind, string> = {
  success: 'border-emerald-400/30 bg-emerald-950/90 text-emerald-50',
  error: 'border-red-400/30 bg-red-950/90 text-red-50',
  info: 'border-cyan-400/30 bg-slate-950/95 text-cyan-50',
}

export function Toaster() {
  const toasts = useAppStore((state) => state.toasts)
  const removeToast = useAppStore((state) => state.removeToast)

  useEffect(() => {
    const timers = toasts.map((toast) => window.setTimeout(() => removeToast(toast.id), 3600))
    return () => timers.forEach(window.clearTimeout)
  }, [removeToast, toasts])

  return (
    <div className="fixed right-5 top-20 z-50 flex w-[min(420px,calc(100vw-40px))] flex-col gap-3">
      {toasts.map((toast) => {
        const Icon = icons[toast.kind]
        return (
          <div key={toast.id} className={cn('rounded-xl border p-4 shadow-2xl backdrop-blur', tones[toast.kind])}>
            <div className="flex items-start gap-3">
              <Icon className="mt-0.5 h-5 w-5 shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold">{toast.title}</p>
                {toast.description ? <p className="mt-1 text-xs leading-5 opacity-80">{toast.description}</p> : null}
              </div>
              <button
                aria-label="关闭提示"
                className="rounded-md p-1 opacity-70 hover:bg-white/10 hover:opacity-100"
                onClick={() => removeToast(toast.id)}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
