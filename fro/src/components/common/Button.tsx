import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  icon?: ReactNode
}

const variants: Record<ButtonVariant, string> = {
  primary:
    'bg-gradient-to-r from-cyan-300 to-sky-500 text-slate-950 shadow-[0_0_26px_rgba(14,165,233,0.28)] hover:from-cyan-200 hover:to-sky-400 active:from-cyan-400 active:to-sky-600',
  secondary: 'border border-sky-400/36 bg-sky-400/10 text-cyan-50 hover:bg-sky-400/16',
  ghost: 'text-slate-200 hover:bg-white/[0.07]',
  danger: 'border border-red-400/42 bg-red-500/10 text-red-100 hover:bg-red-500/18',
}

export function Button({ className, variant = 'primary', icon, children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex min-h-10 items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50',
        variants[variant],
        className,
      )}
      {...props}
    >
      {icon}
      {children}
    </button>
  )
}
