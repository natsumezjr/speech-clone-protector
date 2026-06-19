import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  icon?: ReactNode
}

const variants: Record<ButtonVariant, string> = {
  primary:
    'bg-cyan-400 text-slate-950 shadow-[0_0_24px_rgba(34,211,238,0.24)] hover:bg-cyan-300 active:bg-cyan-500',
  secondary: 'border border-cyan-400/30 bg-cyan-400/10 text-cyan-100 hover:bg-cyan-400/18',
  ghost: 'text-slate-200 hover:bg-white/8',
  danger: 'border border-red-400/40 bg-red-500/10 text-red-100 hover:bg-red-500/18',
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
